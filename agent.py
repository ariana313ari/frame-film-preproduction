"""FRAME orchestrator.

One user turn =
  analyse -> decide (user action / keywords / model proposal) -> VERIFY the proposal
  -> ask for missing info -> check prerequisites -> execute registered tool(s)
  -> validate -> update the Project (only with valid results) -> explain + next step.

Hard bounds: at most `MAX_TOOL_CALLS` tool executions per turn and no recursion, so
the workflow cannot loop. The model may *propose* an action; Python decides.
No hidden reasoning is shown: only short status messages.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

import prompts
from llm_provider import InvalidModelOutputError, LLMError, LLMProvider, generate_validated
from schemas import (
    ArtifactStatus,
    Intent,
    Project,
    RouterProposal,
    Stage,
    ToolResult,
)
from tools import INTENT_TO_TOOL, TOOL_REGISTRY, ToolSpec, describe_tools

MAX_TOOL_CALLS = 4

STATUS_TEXT = {
    "analyze": "Төслийг шинжилж байна",
    "verify": "Үр дүнг шалгаж байна",
    "develop_idea": "Санааг хөгжүүлж байна",
    "build_script": "Сценар бичиж байна",
    "breakdown_scenes": "Дүр зургийг задалж байна",
    "create_shot_list": "Шотын жагсаалт үүсгэж байна",
    "estimate_resources": "Нөөц, хуваарийг тооцоолж байна",
    "check_continuity": "Тасралтгүй байдлыг шалгаж байна",
    "export_package": "Үйлдвэрлэлийн багцыг бэлтгэж байна",
    "general": "Хариулт бэлтгэж байна",
    "route": "Хүсэлтийг ойлгож байна",
}

NEXT_LABEL = {
    Intent.DEVELOP_IDEA: "Санаагаа хөгжүүлэх",
    Intent.BUILD_SCRIPT: "Сценар бичих",
    Intent.BREAKDOWN_SCENES: "Дүр зургийг задлах",
    Intent.CREATE_SHOT_LIST: "Шотын жагсаалт үүсгэх",
    Intent.PLAN_PRODUCTION: "Үйлдвэрлэлийн төлөвлөгөө гаргах",
    Intent.CHECK_CONTINUITY: "Тасралтгүй байдлыг шалгах",
    Intent.EXPORT_PACKAGE: "Багцыг экспортлох",
}

STAGE_LABEL = {
    Stage.IDEA: "санаа", Stage.SCRIPT: "сценар", Stage.SCENES: "дүр зургийн задлал",
    Stage.SHOTS: "шотын жагсаалт", Stage.PLAN: "үйлдвэрлэлийн төлөвлөгөө", Stage.VALIDATION: "шалгалт",
}

PIPELINE_STAGES = (Stage.SCRIPT, Stage.SCENES, Stage.SHOTS, Stage.PLAN, Stage.VALIDATION)
STAGE_TO_TOOL = {spec.stage: spec.name for spec in TOOL_REGISTRY.values() if spec.stage}

# --------------------------------------------------------------------------- #
# Keyword intent detection (fast path; no model call needed)
# --------------------------------------------------------------------------- #
_KEYWORDS: list[tuple[Intent, tuple[str, ...]]] = [  # order = tie-break priority
    (Intent.EXPORT_PACKAGE, ("export", "экспорт", "татаж", "татах", "download", "markdown", "багц", "package")),
    (Intent.CHECK_CONTINUITY, ("continuity", "validat", "шалга", "тасралт", "нийцэл", "зөрчил", "consisten", "check")),
    (Intent.PLAN_PRODUCTION, ("production plan", "resource", "schedule", "төлөвл", "хуваарь", "нөөц",
                              "тоног төхөөрөмж", "equipment", "props", "үйлдвэрлэлийн")),
    (Intent.CREATE_SHOT_LIST, ("shot", "шот", "зураглал", "storyboard")),
    (Intent.BREAKDOWN_SCENES, ("breakdown", "break down", "scene", "дүр зураг", "дүр зург", "задал", "задла", "задлан")),
    (Intent.BUILD_SCRIPT, ("script", "сценар", "логлайн", "logline", "синопсис", "synopsis", "screenplay")),
    (Intent.DEVELOP_IDEA, ("idea", "санаа", "хөгжүүл", "brainstorm", "develop", "concept")),
]
_ACTION_MARKERS = ("үүсгэ", "хий", "бич", "гарга", "зохио", "сэргээ", "дахин", "create", "make", "generate",
                   "build", "write", "run", "export", "check", "validate", "break", "plan", "develop",
                   "шалга", "задал", "задла", "татаж", "төлөвл", "хөгжүүл", "regenerate", "redo")
_PIPELINE_MARKERS = ("бүгдийг", "бүх ", "бүрэн", "full", "everything", "end-to-end", "end to end",
                     "all steps", "complete package", "дараалан", "автоматаар")


def detect_intent(message: str) -> Intent | None:
    """Keyword routing. Returns None when the message is not clearly an action request."""
    text = message.lower()
    if ("?" in text or "？" in text) and not any(m in text for m in _ACTION_MARKERS):
        return None
    best: tuple[int, Intent] | None = None
    for intent, words in _KEYWORDS:
        score = sum(1 for w in words if w in text)
        if score and (best is None or score > best[0]):
            best = (score, intent)
    return best[1] if best else None


def is_pipeline_request(message: str) -> bool:
    text = message.lower()
    return any(m in text for m in _PIPELINE_MARKERS)


def next_intent(project: Project) -> Intent:
    """The most useful next step given the current project state."""
    if not project.idea.strip() and project.get_status(Stage.SCRIPT) == ArtifactStatus.EMPTY:
        return Intent.DEVELOP_IDEA
    for stage in PIPELINE_STAGES:
        if not project.is_fresh(stage):
            return TOOL_REGISTRY[STAGE_TO_TOOL[stage]].intent
    return Intent.EXPORT_PACKAGE


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class StepRecord:
    tool: str
    ok: bool
    message: str
    elapsed_sec: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class AgentTurn:
    reply: str
    intent: Intent
    ok: bool = True
    steps: list[StepRecord] = field(default_factory=list)
    status_log: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    suggested_intent: Intent | None = None
    export_markdown: str | None = None
    export_filename: str | None = None


@dataclass
class _Decision:
    intent: Intent
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    pipeline: bool = False
    error: str | None = None
    source: str = "keyword"


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class FrameAgent:
    def __init__(self, llm: LLMProvider | None, registry: dict[str, ToolSpec] | None = None,
                 max_tool_calls: int = MAX_TOOL_CALLS):
        self.llm = llm
        self.registry = registry if registry is not None else TOOL_REGISTRY
        self.max_tool_calls = max_tool_calls
        self._llm_status = None  # cached per turn

    # -- model availability ------------------------------------------------ #
    def _llm_ready(self) -> bool:
        if self.llm is None:
            return False
        if self._llm_status is None:
            self._llm_status = self.llm.check_status()
        return self._llm_status.available

    def _llm_problem(self) -> str:
        if self.llm is None:
            return "LLM загвар тохируулагдаагүй байна."
        self._llm_ready()
        return self._llm_status.message

    # -- public entry point -------------------------------------------------- #
    def handle(self, project: Project, message: str, *, forced_intent: Intent | None = None,
               forced_args: dict[str, Any] | None = None, history: list[dict] | None = None,
               on_status: Callable[[str], None] | None = None) -> AgentTurn:
        """Process one user turn. Mutates `project` only with validated tool results."""
        self._llm_status = None
        log: list[str] = []

        def status(text: str) -> None:
            log.append(text)
            if on_status:
                on_status(text)

        message = (message or "").strip()
        status(STATUS_TEXT["analyze"])

        decision = self._decide(project, message, forced_intent, forced_args, status)
        if decision.error:
            return AgentTurn(reply=f"⚠️ {decision.error}", intent=decision.intent, ok=False,
                             status_log=log, suggested_intent=next_intent(project))

        if decision.intent == Intent.GENERAL_QUESTION:
            return self._answer_question(project, message, history or [], status, log)

        return self._run_plan(project, decision, message, status, log)

    # -- decide ---------------------------------------------------------------- #
    def _decide(self, project, message, forced_intent, forced_args, status) -> _Decision:
        if forced_intent is not None:
            return self._decision_from_intent(project, forced_intent, forced_args or {}, "user_action")

        intent = detect_intent(message)
        if intent is not None:
            args: dict[str, Any] = {}
            spec = self.registry[INTENT_TO_TOOL[intent]]
            if spec.stage and project.get_status(spec.stage) != ArtifactStatus.EMPTY and "instructions" in spec.args_model.model_fields:
                args["instructions"] = message  # a change request on an existing artifact
            if (intent in (Intent.DEVELOP_IDEA, Intent.BUILD_SCRIPT) and not project.idea.strip()
                    and len(message) >= 30):
                project.idea = message  # first message doubles as the premise
            decision = self._decision_from_intent(project, intent, args, "keyword")
            decision.pipeline = is_pipeline_request(message)
            return decision

        if not project.idea.strip() and "?" not in message and len(message) >= 15:
            project.idea = message  # no project yet: treat the message as the film idea
            return self._decision_from_intent(project, Intent.DEVELOP_IDEA, {}, "implicit_idea")

        return self._ask_model_to_route(project, message, status)

    def _decision_from_intent(self, project, intent: Intent, args: dict, source: str) -> _Decision:
        if intent == Intent.GENERAL_QUESTION:
            return _Decision(intent=intent, source=source)
        tool = INTENT_TO_TOOL.get(intent)
        return self._verified(_Decision(intent=intent, tool=tool, arguments=args, source=source))

    def _ask_model_to_route(self, project: Project, message: str, status) -> _Decision:
        if not self._llm_ready():
            return _Decision(
                intent=Intent.GENERAL_QUESTION,
                error=("Хүсэлтийг ойлгохын тулд загвар хэрэгтэй, гэвч одоогоор боломжгүй: "
                       f"{self._llm_problem()} Загваргүйгээр «шалгах», «төлөвлөгөө», «экспорт» гэх мэт "
                       "тодорхой командууд ажиллана."),
                source="model")
        status(STATUS_TEXT["route"])
        try:
            proposal = generate_validated(
                self.llm,
                prompts.router_messages(user_message=message, project_summary=project.context_summary(),
                                        tools_description=describe_tools()),
                RouterProposal, temperature=0.1)
        except InvalidModelOutputError:
            status("Загвар хүсэлтийг тодорхойлж чадсангүй — ерөнхий хариулт горимд шилжлээ")
            return _Decision(intent=Intent.GENERAL_QUESTION, source="model_fallback")
        except LLMError as exc:
            return _Decision(intent=Intent.GENERAL_QUESTION, error=str(exc), source="model")

        if proposal.intent == Intent.GENERAL_QUESTION and not proposal.tool:
            return _Decision(intent=Intent.GENERAL_QUESTION, source="model")
        decision = _Decision(intent=proposal.intent, tool=proposal.tool or INTENT_TO_TOOL.get(proposal.intent),
                             arguments=proposal.arguments, source="model")
        return self._verified(decision)

    def _verified(self, decision: _Decision) -> _Decision:
        """The orchestrator, not the model, decides what is allowed to run."""
        if decision.tool not in self.registry:
            decision.error = (f"Загвар '{decision.tool}' гэсэн tool санал болголоо, гэвч ийм tool бүртгэлд "
                              f"байхгүй тул ажиллуулсангүй. Боломжит tool-ууд: {', '.join(self.registry)}.")
            return decision
        spec = self.registry[decision.tool]
        decision.intent = spec.intent
        try:
            decision.arguments = spec.args_model.model_validate(decision.arguments).model_dump()
        except ValidationError as exc:
            decision.error = (f"'{spec.name}' tool-д өгсөн аргумент буруу ({exc.errors()[0]['msg']}: "
                              f"{exc.errors()[0]['loc']}) тул ажиллуулсангүй.")
        return decision

    # -- plan + execute ----------------------------------------------------------- #
    def _plan_tools(self, project: Project, decision: _Decision) -> list[tuple[ToolSpec, dict]]:
        spec = self.registry[decision.tool]
        if not decision.pipeline:
            return [(spec, decision.arguments)]
        start = PIPELINE_STAGES.index(spec.stage) if spec.stage in PIPELINE_STAGES else 0
        plan = []
        for stage in PIPELINE_STAGES[start:]:
            tool_name = STAGE_TO_TOOL[stage]
            if tool_name not in self.registry:
                continue
            if stage == spec.stage or not project.is_fresh(stage):
                plan.append((self.registry[tool_name], decision.arguments if stage == spec.stage else {}))
        return plan

    @staticmethod
    def _missing_info(project: Project, spec: ToolSpec) -> list[str]:
        questions = []
        if spec.name in ("develop_idea", "build_script") and not project.idea.strip():
            questions.append("Киноны санаагаа (premise) 1-3 өгүүлбэрээр бичнэ үү.")
        if spec.name == "build_script" and not project.genre.strip():
            questions.append("Киноны жанр юу вэ? (жишээ нь: сэтгэл түгшээсэн триллер, драм, инээдмийн)")
        return questions

    def _run_plan(self, project, decision, message, status, log) -> AgentTurn:
        planned = self._plan_tools(project, decision)
        steps: list[StepRecord] = []
        parts: list[str] = []
        turn = AgentTurn(reply="", intent=decision.intent, steps=steps, status_log=log)
        executed = 0

        for index, (spec, args) in enumerate(planned):
            # 1. missing information -> ask, do not guess
            questions = self._missing_info(project, spec)
            if questions:
                parts.append("Үргэлжлүүлэхийн өмнө надад нэмэлт мэдээлэл хэрэгтэй байна:\n" +
                             "\n".join(f"- {q}" for q in questions))
                turn.needs_clarification = True
                break
            # 2. prerequisites
            blocker = self._unmet_prerequisite(project, spec)
            if blocker:
                parts.append(blocker[0])
                turn.suggested_intent = blocker[1]
                turn.ok = bool(steps)
                break
            # 3. model availability for LLM tools
            if spec.needs_llm and not self._llm_ready():
                parts.append(f"❌ «{spec.name}» нь AI загвар шаардана, гэвч боломжгүй байна.\n"
                             f"{self._llm_problem()}\n\nЗагваргүй (demo) горимд «шалгах», «төлөвлөгөө», "
                             "«экспорт» ажиллана.")
                turn.ok = False
                break
            # 4. hard bound on tool executions
            if executed >= self.max_tool_calls:
                remaining = ", ".join(s.name for s, _ in planned[index:])
                parts.append(f"Нэг хүсэлтэд дээд тал нь {self.max_tool_calls} tool ажиллуулдаг тул зогслоо. "
                             f"Үлдсэн алхам: {remaining}.")
                break

            executed += 1
            result, record = self._execute(project, spec, args, status)
            steps.append(record)
            if not result.ok:
                parts.append(f"❌ «{spec.name}» амжилтгүй боллоо: {result.error}\n"
                             "Төслийн өгөгдөл өөрчлөгдсөнгүй.")
                turn.ok = False
                break
            parts.append(record.message)
            if spec.name == "export_package":
                turn.export_markdown = result.data["markdown"]
                turn.export_filename = result.data["filename"]

        if not parts:
            parts.append("Хийх зүйл олдсонгүй.")
        if turn.suggested_intent is None and not turn.needs_clarification and turn.ok:
            turn.suggested_intent = next_intent(project)
        if turn.suggested_intent is not None:
            parts.append(f"**Дараагийн алхам:** {NEXT_LABEL[turn.suggested_intent]}.")
        turn.reply = "\n\n".join(parts)
        return turn

    def _unmet_prerequisite(self, project: Project, spec: ToolSpec) -> tuple[str, Intent] | None:
        for stage in spec.prerequisites:
            status = project.get_status(stage)
            if status == ArtifactStatus.FRESH:
                continue
            producer = self.registry[STAGE_TO_TOOL[stage]]
            if status == ArtifactStatus.OUTDATED:
                text = (f"«{spec.name}»-ийн өмнө {STAGE_LABEL[stage]}-г шинэчлэх хэрэгтэй: "
                        "дээд түвшний өгөгдөл өөрчлөгдсөн тул хуучирсан байна.")
            else:
                text = f"«{spec.name}»-ийн өмнө {STAGE_LABEL[stage]} хэрэгтэй, одоогоор үүсээгүй байна."
            return text, producer.intent
        return None

    def _execute(self, project: Project, spec: ToolSpec, args: dict, status) -> tuple[ToolResult, StepRecord]:
        status(STATUS_TEXT.get(spec.name, spec.name))
        started = time.monotonic()
        try:
            validated = spec.args_model.model_validate(args)
            llm = self.llm if self._llm_ready() else None
            result = spec.func(**spec.build_kwargs(project, validated, llm))
            if not isinstance(result, ToolResult):
                result = ToolResult.failure(spec.name, "Tool буруу бүтэцтэй үр дүн буцаалаа.")
        except Exception as exc:  # a tool must never crash the app
            result = ToolResult.failure(spec.name, f"Гэнэтийн алдаа ({type(exc).__name__}): {exc}")

        if result.ok:
            status(STATUS_TEXT["verify"])
            try:
                spec.apply(project, result)  # Pydantic re-validates the payload here
            except (ValidationError, ValueError, KeyError) as exc:
                result = ToolResult.failure(spec.name, f"Үр дүн шалгалтад тэнцсэнгүй: {exc}")

        elapsed = time.monotonic() - started
        message = self._summarize(project, spec, result) if result.ok else (result.error or "")
        return result, StepRecord(tool=spec.name, ok=result.ok, message=message, elapsed_sec=elapsed,
                                  warnings=list(result.warnings))

    # -- explanations (deterministic: no model hallucination in status reports) --- #
    def _summarize(self, project: Project, spec: ToolSpec, result: ToolResult) -> str:
        name = spec.name
        if name == "develop_idea":
            text = (f"✅ Санааг хөгжүүллээ: {len(project.idea_notes.logline_options)} логлайны санал, "
                    f"{len(project.idea_notes.story_questions)} тодруулах асуулт. "
                    "«Idea & Story» таб дээрээс хараад шийдвэрүүдээ батална уу.")
        elif name == "build_script":
            text = (f"✅ Сценар бэлэн боллоо: {len(project.script_scenes)} дүр зураг, "
                    f"{len(project.characters)} дүр.\n\n*Логлайн:* {project.logline}")
        elif name == "breakdown_scenes":
            places = {s.location.casefold() for s in project.scenes}
            text = (f"✅ Дүр зургийг задаллаа: {len(project.scenes)} дүр зураг, нийт "
                    f"{project.total_scene_seconds:g} сек (зорилт {project.duration_seconds:g} сек), "
                    f"{len(places)} байршил.")
        elif name == "create_shot_list":
            total = sum(s.estimated_duration_sec for s in project.shots)
            text = f"✅ Шотын жагсаалт бэлэн: {len(project.shots)} шот, нийт ~{total:g} сек."
        elif name == "estimate_resources":
            plan = project.plan
            text = (f"✅ Төлөвлөгөө гарлаа (таамаглал): {len(plan.props)} prop, {len(plan.equipment)} "
                    f"төрлийн тоног төхөөрөмж, {len(plan.locations)} байршил, ~{plan.total_minutes} мин / "
                    f"{plan.shoot_days} өдөр. Таамаглал, хязгаарлалтыг табын доор тэмдэглэв.")
        elif name == "check_continuity":
            counts = result.data.get("counts", {})
            model_n = sum(1 for i in result.issues if i.source == "model")
            text = (f"✅ Шалгалт дууслаа: {counts.get('ERROR', 0)} алдаа, {counts.get('WARNING', 0)} анхааруулга, "
                    f"{counts.get('INFO', 0)} тэмдэглэл. Үүнээс {model_n} нь AI-ийн санал (баталгаагүй), "
                    "бусад нь дүрэмд суурилсан (deterministic).")
        elif name == "export_package":
            text = (f"✅ Үйлдвэрлэлийн багц бэлэн ({result.data['characters']} тэмдэгт, Markdown). "
                    "«Validation» таб дээрээс татаж авна уу.")
        else:
            text = f"✅ {name} дууслаа."

        outdated = [STAGE_LABEL[s] for s in Stage
                    if project.get_status(s) == ArtifactStatus.OUTDATED]
        if outdated and spec.stage in (Stage.SCRIPT, Stage.SCENES, Stage.SHOTS):
            text += f"\n\n⚠️ Хуучирсан болсон: {', '.join(outdated)}. Дахин үүсгэх хүртэл найдахгүй байна уу."
        if result.warnings:
            text += "\n\n" + "\n".join(f"- ⚠️ {w}" for w in result.warnings[:6])
        return text

    # -- general questions -------------------------------------------------------- #
    def _answer_question(self, project, message, history, status, log) -> AgentTurn:
        if not self._llm_ready():
            return AgentTurn(
                reply=("❌ Асуултад хариулахад AI загвар хэрэгтэй, гэвч боломжгүй байна.\n"
                       f"{self._llm_problem()}"),
                intent=Intent.GENERAL_QUESTION, ok=False, status_log=log,
                suggested_intent=next_intent(project))
        status(STATUS_TEXT["general"])
        try:
            answer = self.llm.chat(prompts.general_answer_messages(
                project_summary=project.context_summary(), history=history, user_message=message,
                language=project.language))
        except LLMError as exc:
            return AgentTurn(reply=f"❌ {exc}", intent=Intent.GENERAL_QUESTION, ok=False, status_log=log)
        suggested = next_intent(project)
        reply = f"{answer.strip()}\n\n**Дараагийн алхам:** {NEXT_LABEL[suggested]}."
        return AgentTurn(reply=reply, intent=Intent.GENERAL_QUESTION, status_log=log,
                         suggested_intent=suggested)
