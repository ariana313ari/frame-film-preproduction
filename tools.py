"""Real Python tools executed by the orchestrator.

Every tool:
  * validates its inputs (`ToolInputError` -> failed ToolResult),
  * returns a `ToolResult` (never raises for expected problems),
  * only returns data that already passed Pydantic validation.

Tools do not touch the Project; the registry's `apply` functions do that after the
orchestrator has verified the result.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import prompts
from export_utils import project_to_markdown, safe_filename
from llm_provider import (
    InvalidModelOutputError,
    LLMError,
    LLMProvider,
    generate_validated,
)
from schemas import (
    CameraMovement,
    Character,
    EquipmentItem,
    IdeaNotes,
    Intent,
    LocationItem,
    ProductionPlan,
    Project,
    PropItem,
    Scene,
    SceneBreakdownOutput,
    ScheduleBlock,
    ScriptOutput,
    ScriptScene,
    SemanticReviewOutput,
    Severity,
    Shot,
    ShotListOutput,
    Stage,
    TimeOfDay,
    ToolResult,
    ValidationIssue,
)


class ToolInputError(ValueError):
    """Invalid or missing tool input."""


def tool_guard(name: str):
    """Turn expected failures into a failed ToolResult with a readable message."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> ToolResult:
            try:
                return func(*args, **kwargs)
            except ToolInputError as exc:
                return ToolResult.failure(name, f"Оролт буруу: {exc}")
            except InvalidModelOutputError as exc:
                return ToolResult.failure(name, str(exc))
            except LLMError as exc:
                return ToolResult.failure(name, str(exc))
            except ValidationError as exc:
                return ToolResult.failure(name, f"Өгөгдөл шалгалтад унав: {exc.error_count()} алдаа ({exc.errors()[0]['msg']})")

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def _require_text(label: str, value: str | None) -> str:
    if not value or not str(value).strip():
        raise ToolInputError(f"'{label}' хоосон байна.")
    return str(value).strip()


def _require_duration(minutes: float) -> float:
    try:
        minutes = float(minutes)
    except (TypeError, ValueError):
        raise ToolInputError("Хугацаа тоо байх ёстой.") from None
    if not 0 < minutes <= 60:
        raise ToolInputError("Хугацаа 0-ээс их, 60 минутаас бага байх ёстой.")
    return minutes


def _require_llm(llm: LLMProvider | None) -> LLMProvider:
    if llm is None:
        raise ToolInputError("Энэ tool-д LLM загвар шаардлагатай, гэвч холбогдоогүй байна (demo горим).")
    return llm


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _origin(llm: LLMProvider | None) -> str:
    return f"model:{llm.model}" if llm else "deterministic"


# --------------------------------------------------------------------------- #
# Deterministic validation rules (shared by check_continuity, export_package
# and the generation tools' post-checks)
# --------------------------------------------------------------------------- #
DURATION_WARN_RATIO = 0.15   # total scene time vs requested film duration
DURATION_ERROR_RATIO = 0.40
SHOT_SCENE_WARN_RATIO = 0.35  # sum of shots vs scene duration

_EQUIPMENT_FOR_MOVEMENT = {
    CameraMovement.DOLLY: ("dolly", "slider", "track", "долли"),
    CameraMovement.TRACKING: ("dolly", "slider", "track", "gimbal", "steadicam", "долли"),
    CameraMovement.GIMBAL: ("gimbal", "steadicam", "stabiliz", "гимбал"),
    CameraMovement.CRANE: ("crane", "jib", "кран"),
}


def run_deterministic_checks(
    *,
    characters: Sequence[Character],
    scenes: Sequence[Scene],
    shots: Sequence[Shot],
    duration_minutes: float,
    max_locations: int | None = None,
    max_actors: int | None = None,
    equipment: Sequence[str] = (),
) -> list[ValidationIssue]:
    """Rule-based checks. Everything returned here has source='deterministic'."""
    issues: list[ValidationIssue] = []

    def add(severity, code, message, suggestion="", scene=None, shot=None):
        issues.append(ValidationIssue(severity=severity, code=code, message=message,
                                      suggestion=suggestion, scene_number=scene, shot_number=shot))

    known = {_norm(c.name): c.name for c in characters}
    scene_numbers = [s.number for s in scenes]
    scene_set = set(scene_numbers)

    # -- scene numbering -------------------------------------------------- #
    dupes = sorted({n for n in scene_numbers if scene_numbers.count(n) > 1})
    for n in dupes:
        add(Severity.ERROR, "SCENE_NUMBER_DUPLICATE", f"Дүр зургийн дугаар {n} давхардсан байна.",
            "Дүр зураг бүрт өвөрмөц дугаар өгнө үү.", scene=n)
    if scene_set and sorted(scene_set) != list(range(1, len(scene_set) + 1)):
        add(Severity.WARNING, "SCENE_NUMBER_GAP", "Дүр зургийн дугаар 1-ээс завсаргүй дараалаагүй байна.",
            "Дугаарыг 1, 2, 3 ... гэж дахин дугаарлана уу.")

    # -- scene durations -------------------------------------------------- #
    for s in scenes:
        if s.estimated_duration_sec <= 0:
            add(Severity.ERROR, "SCENE_DURATION_INVALID",
                f"Дүр зураг {s.number}-ийн хугацаа эерэг тоо биш ({s.estimated_duration_sec:g} сек).",
                "Тухайн дүр зурагт 0-ээс их секунд онооно уу.", scene=s.number)
    if scenes and all(s.estimated_duration_sec > 0 for s in scenes):
        total, target = sum(s.estimated_duration_sec for s in scenes), duration_minutes * 60
        ratio = abs(total - target) / target
        if ratio > DURATION_WARN_RATIO:
            severity = Severity.ERROR if ratio > DURATION_ERROR_RATIO else Severity.WARNING
            direction = "урт" if total > target else "богино"
            add(severity, "FILM_DURATION_MISMATCH",
                f"Дүр зургуудын нийт хугацаа {total:g} сек, зорилтот {target:g} сек-ээс "
                f"{ratio:.0%}-иар {direction} байна.",
                "Дүр зургийн хугацааг тохируулах эсвэл зорилтот хугацааг өөрчилнө үү.")

    # -- characters ---------------------------------------------------------- #
    used: set[str] = set()
    for s in scenes:
        for name in s.characters:
            key = _norm(name)
            if key in known:
                used.add(key)
            else:
                add(Severity.ERROR, "CHARACTER_UNKNOWN",
                    f"Дүр зураг {s.number}-д '{name}' дүр дурдагдсан ч дүрүүдийн жагсаалтад алга.",
                    "Дүрийг жагсаалтад нэмэх эсвэл дүр зургийн нэрийг засна уу.", scene=s.number)
    if scenes:
        for key, name in known.items():
            if key not in used:
                add(Severity.INFO, "CHARACTER_UNUSED", f"'{name}' дүр ямар ч дүр зурагт ороогүй байна.",
                    "Дүрийг хасах эсвэл дүр зурагт оруулна уу.")

    # -- locations & actors vs project constraints ------------------------ #
    locations = {_norm(s.location): s.location for s in scenes}
    if max_locations and len(locations) > max_locations:
        add(Severity.ERROR, "LOCATION_LIMIT",
            f"{len(locations)} өөр байршил байна ({', '.join(locations.values())}); хязгаар {max_locations}.",
            "Байршлуудыг нэгтгэх эсвэл хязгаарыг өөрчилнө үү.")
    if max_actors:
        on_screen = {_norm(n) for s in scenes for n in s.characters}
        if len(on_screen) > max_actors:
            add(Severity.ERROR, "ACTOR_LIMIT",
                f"Нийт {len(on_screen)} дүр дэлгэцэнд гарч байна; жүжигчний хязгаар {max_actors}.",
                "Жижиг дүрүүдийг нэгтгэх эсвэл хасна уу.")
        for s in scenes:
            if len({_norm(n) for n in s.characters}) > max_actors:
                add(Severity.ERROR, "ACTOR_LIMIT_SCENE",
                    f"Дүр зураг {s.number}-д {len(s.characters)} дүр байна; хязгаар {max_actors}.",
                    "Тухайн дүр зургаас дүр хасна уу.", scene=s.number)

    # -- time of day --------------------------------------------------------- #
    for s in scenes:
        if s.time_of_day == TimeOfDay.UNSPECIFIED:
            add(Severity.INFO, "TIME_OF_DAY_MISSING", f"Дүр зураг {s.number}-ийн цаг хугацаа заагаагүй.",
                "Өдөр/шөнө гэх мэтийг зааж өгвөл гэрэлтүүлгийн төлөвлөгөө тодорхой болно.", scene=s.number)

    # -- shots --------------------------------------------------------------- #
    if shots:
        shot_numbers = [sh.number for sh in shots]
        for n in sorted({n for n in shot_numbers if shot_numbers.count(n) > 1}):
            add(Severity.ERROR, "SHOT_NUMBER_DUPLICATE", f"Шот №{n} давхардсан байна.",
                "Шотуудыг 1, 2, 3 ... гэж дахин дугаарлана уу.", shot=n)
        if sorted(set(shot_numbers)) != list(range(1, len(set(shot_numbers)) + 1)):
            add(Severity.WARNING, "SHOT_NUMBER_GAP", "Шотын дугаар 1-ээс завсаргүй дараалаагүй байна.",
                "Шотуудыг дахин дугаарлана уу.")

        per_scene: dict[int, float] = {}
        for sh in shots:
            if sh.scene_number not in scene_set:
                add(Severity.ERROR, "SHOT_SCENE_MISSING",
                    f"Шот №{sh.number} дүр зураг {sh.scene_number}-д холбогдсон ч ийм дүр зураг байхгүй.",
                    "Шотыг байгаа дүр зурагт холбох эсвэл устгана уу.", shot=sh.number)
            if sh.estimated_duration_sec <= 0:
                add(Severity.ERROR, "SHOT_DURATION_INVALID",
                    f"Шот №{sh.number}-ийн хугацаа эерэг тоо биш ({sh.estimated_duration_sec:g} сек).",
                    "0-ээс их секунд онооно уу.", scene=sh.scene_number, shot=sh.number)
            else:
                per_scene[sh.scene_number] = per_scene.get(sh.scene_number, 0) + sh.estimated_duration_sec

        covered = {sh.scene_number for sh in shots}
        for s in scenes:
            if s.number not in covered:
                add(Severity.WARNING, "SCENE_WITHOUT_SHOTS", f"Дүр зураг {s.number}-д шот байхгүй байна.",
                    "Энэ дүр зурагт шот нэмнэ үү.", scene=s.number)
            elif s.estimated_duration_sec > 0 and s.number in per_scene:
                ratio = abs(per_scene[s.number] - s.estimated_duration_sec) / s.estimated_duration_sec
                if ratio > SHOT_SCENE_WARN_RATIO:
                    add(Severity.WARNING, "SHOT_SCENE_DURATION_MISMATCH",
                        f"Дүр зураг {s.number}: шотуудын нийлбэр {per_scene[s.number]:g} сек, "
                        f"дүр зургийн хугацаа {s.estimated_duration_sec:g} сек.",
                        "Шотын эсвэл дүр зургийн хугацааг тааруулна уу.", scene=s.number)

        # Equipment vs camera movement (only when the user listed equipment)
        gear = [_norm(e) for e in equipment]
        if gear:
            for sh in shots:
                needed = _EQUIPMENT_FOR_MOVEMENT.get(sh.movement)
                if needed and not any(any(k in g for k in needed) for g in gear):
                    add(Severity.WARNING, "EQUIPMENT_MISSING",
                        f"Шот №{sh.number} {sh.movement.value} хөдөлгөөнтэй ч тоног төхөөрөмжийн жагсаалтад "
                        "тохирох хэрэгсэл алга.",
                        "Хөдөлгөөнийг STATIC/HANDHELD болгох эсвэл хэрэгслийг жагсаалтад нэмнэ үү.",
                        scene=sh.scene_number, shot=sh.number)

        # Props: heuristic text matching (labelled as a heuristic in the message)
        all_props = {_norm(p): p for s in scenes for p in s.props}
        props_in_scene = {s.number: {_norm(p) for p in s.props} for s in scenes}
        for sh in shots:
            text = _norm(f"{sh.description} {sh.audio_notes}")
            for key, label in all_props.items():
                if key and key in text and sh.scene_number in props_in_scene and \
                        key not in props_in_scene[sh.scene_number]:
                    add(Severity.INFO, "PROP_NOT_LISTED",
                        f"Шот №{sh.number} '{label}'-г дурдсан ч энэ дүр зургийн props жагсаалтад байхгүй "
                        "(текст тааруулалтад суурилсан таамаглал).",
                        "Дүр зургийн props-д нэмэх эсвэл шотын тайлбарыг засна уу.",
                        scene=sh.scene_number, shot=sh.number)
    elif scenes:
        add(Severity.INFO, "NO_SHOTS", "Шотын жагсаалт хараахан үүсээгүй тул шотын шалгалт хийгдсэнгүй.")

    # Props that vanish and reappear between scenes (heuristic)
    ordered = sorted(scenes, key=lambda s: s.number)
    for key, label in {_norm(p): p for s in ordered for p in s.props}.items():
        present = [key in {_norm(p) for p in s.props} for s in ordered]
        if present.count(True) >= 2:
            first, last = present.index(True), len(present) - 1 - present[::-1].index(True)
            gaps = [ordered[i].number for i in range(first, last) if not present[i]]
            if gaps:
                add(Severity.INFO, "PROP_GAP",
                    f"'{label}' дүр зураг {', '.join(map(str, gaps))}-д байхгүй ч өмнө/хойно нь гарч байна "
                    "(санаатай эсэхийг шалгана уу).",
                    "Хэрэв prop үргэлжилж байгаа бол props жагсаалтад нэмнэ үү.", scene=gaps[0])
    return issues


# --------------------------------------------------------------------------- #
# Tool arguments (validated by the orchestrator before execution)
# --------------------------------------------------------------------------- #
class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InstructionArgs(ToolArgs):
    instructions: str = Field("", max_length=2000)


class CheckArgs(ToolArgs):
    # AI semantic review is opt-in: slower, and its findings are suggestions only.
    use_model: bool = False


class NoArgs(ToolArgs):
    pass


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool_guard("develop_idea")
def develop_idea(llm: LLMProvider | None, *, idea: str, genre: str = "", duration_minutes: float = 3,
                 constraints: str = "", approved_decisions: Sequence[str] = (), language: str = "mn",
                 instructions: str = "") -> ToolResult:
    idea = _require_text("idea", idea)
    duration_minutes = _require_duration(duration_minutes)
    llm = _require_llm(llm)
    notes = generate_validated(
        llm,
        prompts.develop_idea_messages(idea=idea, genre=genre, duration_minutes=duration_minutes,
                                      constraints=constraints, approved_decisions=list(approved_decisions),
                                      language=language, instructions=instructions),
        IdeaNotes,
        validator=lambda n: _need(n.logline_options, "logline_options хоосон байна"),
    )
    return ToolResult.success("develop_idea", notes.model_dump(mode="json"), meta={"origin": _origin(llm)})


def _need(value, message: str) -> None:
    if not value:
        raise ValueError(message)


@tool_guard("build_script")
def build_script(llm: LLMProvider | None, *, idea: str, genre: str, duration_minutes: float,
                 constraints: str = "", approved_decisions: Sequence[str] = (), visual_style: str = "",
                 max_locations: int | None = None, max_actors: int | None = None, language: str = "mn",
                 instructions: str = "", previous_logline: str = "",
                 previous_synopsis: str = "") -> ToolResult:
    idea = _require_text("idea", idea)
    genre = _require_text("genre", genre)
    duration_minutes = _require_duration(duration_minutes)
    llm = _require_llm(llm)

    out = generate_validated(
        llm,
        prompts.build_script_messages(
            idea=idea, genre=genre, duration_minutes=duration_minutes, constraints=constraints,
            approved_decisions=list(approved_decisions), visual_style=visual_style,
            max_locations=max_locations, max_actors=max_actors, language=language,
            instructions=instructions, previous_logline=previous_logline,
            previous_synopsis=previous_synopsis),
        ScriptOutput,
    )
    for i, scene in enumerate(out.scenes, start=1):  # normalise numbering
        scene.number = i

    warnings = []
    if max_actors and len(out.characters) > max_actors:
        warnings.append(f"Сценарт {len(out.characters)} дүр байна; жүжигчний хязгаар {max_actors}.")
    return ToolResult.success("build_script", out.model_dump(mode="json"), warnings=warnings,
                              meta={"origin": _origin(llm)})


@tool_guard("breakdown_scenes")
def breakdown_scenes(llm: LLMProvider | None, *, logline: str, characters: Sequence[Character],
                     script_scenes: Sequence[ScriptScene], duration_minutes: float,
                     max_locations: int | None = None, max_actors: int | None = None,
                     constraints: str = "", language: str = "mn", instructions: str = "") -> ToolResult:
    if not script_scenes:
        raise ToolInputError("Сценар байхгүй байна. Эхлээд сценар үүсгэнэ үү.")
    duration_minutes = _require_duration(duration_minutes)
    llm = _require_llm(llm)

    def validate(out: SceneBreakdownOutput) -> None:
        bad = [s.number for s in out.scenes if s.estimated_duration_sec <= 0]
        if bad:
            raise ValueError(f"estimated_duration_sec must be a positive number (scenes {bad}).")

    out = generate_validated(
        llm,
        prompts.breakdown_scenes_messages(
            logline=logline, characters=characters, script_scenes=script_scenes,
            duration_minutes=duration_minutes, max_locations=max_locations, max_actors=max_actors,
            constraints=constraints, language=language, instructions=instructions),
        SceneBreakdownOutput,
        validator=validate,
    )
    for i, scene in enumerate(out.scenes, start=1):
        scene.number = i

    found = run_deterministic_checks(characters=characters, scenes=out.scenes, shots=[],
                                     duration_minutes=duration_minutes, max_locations=max_locations,
                                     max_actors=max_actors)
    relevant = [i for i in found if i.severity != Severity.INFO]
    return ToolResult.success(
        "breakdown_scenes", {"scenes": [s.model_dump(mode="json") for s in out.scenes]},
        warnings=[i.message for i in relevant], issues=relevant, meta={"origin": _origin(llm)})


@tool_guard("create_shot_list")
def create_shot_list(llm: LLMProvider | None, *, scenes: Sequence[Scene], characters: Sequence[Character],
                     visual_style: str = "", equipment: Sequence[str] = (), constraints: str = "",
                     language: str = "mn", instructions: str = "") -> ToolResult:
    if not scenes:
        raise ToolInputError("Дүр зургийн задлал байхгүй байна. Эхлээд дүр зургийг задлана уу.")
    llm = _require_llm(llm)
    valid_numbers = {s.number for s in scenes}

    def validate(out: ShotListOutput) -> None:
        unknown = sorted({s.scene_number for s in out.shots} - valid_numbers)
        if unknown:
            raise ValueError(f"scene_number {unknown} does not exist. Valid scene numbers: "
                             f"{sorted(valid_numbers)}.")
        bad = [s.number for s in out.shots if s.estimated_duration_sec <= 0]
        if bad:
            raise ValueError(f"estimated_duration_sec must be positive (shots {bad}).")

    out = generate_validated(
        llm,
        prompts.create_shot_list_messages(
            scenes=scenes, characters=characters, visual_style=visual_style, equipment=list(equipment),
            constraints=constraints, language=language, instructions=instructions),
        ShotListOutput,
        validator=validate,
    )
    shots = sorted(out.shots, key=lambda s: s.scene_number)  # stable: keeps order within a scene
    for i, shot in enumerate(shots, start=1):
        shot.number = i

    warnings = []
    missing = sorted(valid_numbers - {s.scene_number for s in shots})
    if missing:
        warnings.append(f"Дараах дүр зургуудад шот үүссэнгүй: {', '.join(map(str, missing))}.")
    found = run_deterministic_checks(characters=characters, scenes=scenes, shots=shots,
                                     duration_minutes=sum(s.estimated_duration_sec for s in scenes) / 60 or 1,
                                     equipment=equipment)
    relevant = [i for i in found if i.severity == Severity.WARNING or i.severity == Severity.ERROR]
    warnings += [i.message for i in relevant if i.code in {"EQUIPMENT_MISSING", "SHOT_SCENE_DURATION_MISMATCH"}]
    return ToolResult.success("create_shot_list", {"shots": [s.model_dump(mode="json") for s in shots]},
                              warnings=warnings, issues=relevant, meta={"origin": _origin(llm)})


@tool_guard("check_continuity")
def check_continuity(*, characters: Sequence[Character], scenes: Sequence[Scene], shots: Sequence[Shot],
                     duration_minutes: float, max_locations: int | None = None,
                     max_actors: int | None = None, equipment: Sequence[str] = (),
                     llm: LLMProvider | None = None, use_model: bool = True,
                     language: str = "mn") -> ToolResult:
    """Deterministic rule checks, plus optional model-generated semantic suggestions."""
    if not scenes:
        raise ToolInputError("Шалгах дүр зураг байхгүй байна.")
    duration_minutes = _require_duration(duration_minutes)

    issues = run_deterministic_checks(
        characters=characters, scenes=scenes, shots=shots, duration_minutes=duration_minutes,
        max_locations=max_locations, max_actors=max_actors, equipment=equipment)
    origin, warnings = "deterministic", []

    if use_model and llm is not None:
        try:
            review = generate_validated(
                llm, prompts.semantic_review_messages(characters=characters, scenes=scenes,
                                                     shots=shots, language=language),
                SemanticReviewOutput)
            for item in review.issues[:5]:
                issues.append(ValidationIssue(
                    severity=item.severity if item.severity != Severity.ERROR else Severity.WARNING,
                    code="MODEL_REVIEW", message=item.message, suggestion=item.suggestion,
                    source="model", scene_number=item.scene_number))
            origin = f"deterministic+model:{llm.model}"
        except LLMError as exc:
            warnings.append(f"AI-ийн семантик шалгалт хийгдсэнгүй: {exc}")
            issues.append(ValidationIssue(severity=Severity.INFO, code="MODEL_REVIEW_FAILED",
                                          message=f"AI-ийн семантик шалгалт хийгдсэнгүй: {exc}",
                                          source="deterministic"))
    elif use_model:
        warnings.append("LLM холбогдоогүй тул зөвхөн дүрэмд суурилсан (deterministic) шалгалт хийгдлээ.")

    counts = {sev.value: sum(1 for i in issues if i.severity == sev) for sev in Severity}
    return ToolResult.success(
        "check_continuity", {"counts": counts, "total": len(issues)},
        warnings=warnings, issues=issues, meta={"origin": origin})


# -- estimate_resources ------------------------------------------------------- #
SETUP_MINUTES_PER_SHOT = 10     # rule-of-thumb for a small crew: set up + a few takes
MOVEMENT_EXTRA_MINUTES = 5      # extra time for moving-camera shots
BLOCK_LOADIN_MINUTES = 30       # arrive, rig, light the location
BLOCK_WRAP_MINUTES = 15
SHOOT_DAY_MINUTES = 480
_TIME_ORDER = {TimeOfDay.DAWN: 0, TimeOfDay.DAY: 1, TimeOfDay.DUSK: 2, TimeOfDay.NIGHT: 3,
               TimeOfDay.UNSPECIFIED: 4}


def _gear_available(equipment: Sequence[str], keywords: Iterable[str]) -> bool | None:
    gear = [_norm(e) for e in equipment]
    if not gear:
        return None
    return any(any(k in g for k in keywords) for g in gear)


@tool_guard("estimate_resources")
def estimate_resources(*, scenes: Sequence[Scene], shots: Sequence[Shot], equipment: Sequence[str] = (),
                       budget: str = "low", max_actors: int | None = None, language: str = "mn") -> ToolResult:
    """Rule-based resource and schedule estimate. Contains no prices."""
    if not scenes:
        raise ToolInputError("Дүр зургийн задлал байхгүй байна.")
    if not shots:
        raise ToolInputError("Шотын жагсаалт байхгүй байна. Эхлээд шотын жагсаалт үүсгэнэ үү.")

    # props
    prop_scenes: dict[str, tuple[str, list[int]]] = {}
    for s in scenes:
        for p in s.props:
            label, nums = prop_scenes.setdefault(_norm(p), (p, []))
            nums.append(s.number)
    props = [PropItem(name=label, scene_numbers=nums) for label, nums in prop_scenes.values()]

    # locations
    loc_map: dict[str, LocationItem] = {}
    for s in scenes:
        item = loc_map.setdefault(_norm(s.location), LocationItem(name=s.location, time_of_day=s.time_of_day.value))
        item.scene_numbers.append(s.number)
    locations = list(loc_map.values())

    # equipment
    text_all = _norm(" ".join(f"{s.action} {s.location}" for s in scenes) + " " +
                     " ".join(f"{sh.description} {sh.audio_notes}" for sh in shots))
    movements = {sh.movement for sh in shots}
    gear: list[EquipmentItem] = [EquipmentItem(name="Camera", reason="Бүх шотод.",
                                               available=_gear_available(equipment, ("camera", "камер", "phone", "утас")))]
    if CameraMovement.STATIC in movements or not movements - {CameraMovement.STATIC}:
        gear.append(EquipmentItem(name="Tripod", reason="STATIC шотуудад.",
                                  available=_gear_available(equipment, ("tripod", "штатив"))))
    if any(sh.audio_notes.strip() for sh in shots):
        gear.append(EquipmentItem(name="Microphone / sound recorder", reason="Яриа, дууны тэмдэглэлтэй шотууд.",
                                  available=_gear_available(equipment, ("mic", "микроф", "recorder", "zoom"))))
    movement_gear = {
        CameraMovement.DOLLY: ("Dolly / slider (эсвэл орлуулах хэрэгсэл)", ("dolly", "slider", "долли")),
        CameraMovement.TRACKING: ("Tracking rig / gimbal", ("dolly", "slider", "gimbal", "steadicam", "гимбал")),
        CameraMovement.GIMBAL: ("Gimbal / stabilizer", ("gimbal", "steadicam", "stabiliz", "гимбал")),
        CameraMovement.CRANE: ("Crane / jib", ("crane", "jib", "кран")),
        CameraMovement.HANDHELD: ("Shoulder rig (заавал биш)", ("rig", "shoulder")),
    }
    for movement, (name, keys) in movement_gear.items():
        if movement in movements:
            gear.append(EquipmentItem(name=name, reason=f"{movement.value} хөдөлгөөнтэй шотууд.",
                                      available=_gear_available(equipment, keys)))
    if any(s.time_of_day in (TimeOfDay.NIGHT, TimeOfDay.DUSK) for s in scenes):
        gear.append(EquipmentItem(name="Portable lighting / practical lights",
                                  reason="Шөнийн/үдшийн дүр зургууд.",
                                  available=_gear_available(equipment, ("light", "гэрэл", "led"))))
    if re.search(r"rain|бороо|шүүдэр", text_all):
        gear.append(EquipmentItem(name="Rain protection for camera & audio",
                                  reason="Дүр зураг/шотын тайлбарт бороо дурдсан.",
                                  available=_gear_available(equipment, ("rain", "cover", "бороо"))))

    # schedule: group scenes by location, then by time of day
    shot_minutes: dict[int, int] = {}
    for sh in shots:
        extra = MOVEMENT_EXTRA_MINUTES if sh.movement != CameraMovement.STATIC else 0
        shot_minutes[sh.scene_number] = shot_minutes.get(sh.scene_number, 0) + SETUP_MINUTES_PER_SHOT + extra
    groups: dict[tuple[str, TimeOfDay], list[Scene]] = {}
    for s in scenes:
        groups.setdefault((_norm(s.location), s.time_of_day), []).append(s)
    ordered_groups = sorted(groups.items(), key=lambda kv: (
        list(loc_map).index(kv[0][0]), _TIME_ORDER[kv[0][1]]))

    schedule: list[ScheduleBlock] = []
    day, used, order = 1, 0, 0
    for (loc_key, tod), group in ordered_groups:
        minutes = BLOCK_LOADIN_MINUTES + BLOCK_WRAP_MINUTES + sum(shot_minutes.get(s.number, 0) for s in group)
        if used and used + minutes > SHOOT_DAY_MINUTES:
            day, used, order = day + 1, 0, 0
        order += 1
        used += minutes
        schedule.append(ScheduleBlock(
            day=day, order=order, label=f"{loc_map[loc_key].name} · {tod.value}",
            scene_numbers=[s.number for s in group], estimated_minutes=minutes,
            notes=f"{sum(1 for sh in shots if sh.scene_number in {s.number for s in group})} шот"))

    assumptions = [
        f"Шот бүрт ~{SETUP_MINUTES_PER_SHOT} мин (тохируулга + хэдэн дубль), хөдөлгөөнтэй камерт +{MOVEMENT_EXTRA_MINUTES} мин гэж таамагласан.",
        f"Байршил бүрт ачилт/гэрэл ~{BLOCK_LOADIN_MINUTES} мин, цэвэрлэгээ ~{BLOCK_WRAP_MINUTES} мин гэж таамагласан.",
        f"Зураг авалтын өдрийг {SHOOT_DAY_MINUTES // 60} цаг гэж үзсэн.",
        "Эдгээр нь FRAME-ийн ерөнхий дүрмээр гарсан ТААМАГЛАЛ — бодит хугацаа багийн туршлага, цаг агаараас хамаарна.",
        "Хураамж, түрээс, үнэ тооцоогүй (зах зээлийн бодит үнэ оруулаагүй).",
    ]
    if _norm(str(budget)) in ("low", "бага"):
        assumptions.append("Бага төсөв: боломжтой бол байгалийн/практик гэрэл, жижиг баг, байгаа тоног төхөөрөмж ашиглахыг зөвлөнө.")
    if max_actors:
        assumptions.append(f"Жүжигчний хязгаар: {max_actors}.")
    if not equipment:
        assumptions.append("Тоног төхөөрөмжийн жагсаалт өгөөгүй тул 'available' төлөв тодорхойгүй.")

    plan = ProductionPlan(props=props, equipment=gear, locations=locations, schedule=schedule,
                          assumptions=assumptions)
    return ToolResult.success(
        "estimate_resources", plan.model_dump(mode="json"), meta={"origin": "deterministic"},
        warnings=[] if equipment else ["Тоног төхөөрөмжийн жагсаалт хоосон тул боломжит эсэх нь тодорхойгүй."])


@tool_guard("export_package")
def export_package(*, project: Project) -> ToolResult:
    """Render the project as a Markdown production document."""
    if not (project.script_scenes or project.scenes):
        raise ToolInputError("Экспортлох агуулга алга. Хамгийн багадаа сценар эсвэл дүр зургийн задлал хэрэгтэй.")

    warnings = []
    for stage in (Stage.SCRIPT, Stage.SCENES, Stage.SHOTS, Stage.PLAN):
        if project.get_status(stage).value == "outdated":
            warnings.append(f"'{stage.value}' хэсэг хуучирсан (upstream өөрчлөгдсөн) — шинэчлэхийг зөвлөнө.")

    issues, checked = project.issues, project.is_fresh(Stage.VALIDATION)
    if not checked and project.scenes:
        # No fresh validation: attach deterministic findings so the document is never silently unchecked.
        issues = run_deterministic_checks(
            characters=project.characters, scenes=project.scenes,
            shots=project.shots if project.is_fresh(Stage.SHOTS) else [],
            duration_minutes=project.duration_minutes, max_locations=project.max_locations,
            max_actors=project.max_actors, equipment=project.equipment)
        checked = True
        warnings.append("Шалгалт хийгдээгүй/хуучирсан тул экспортод зөвхөн дүрэмд суурилсан шалгалтыг шинээр оруулав.")

    markdown = project_to_markdown(project, issues, checked)
    return ToolResult.success(
        "export_package",
        {"markdown": markdown, "filename": safe_filename(project.name, ".md"), "characters": len(markdown)},
        warnings=warnings, meta={"origin": "deterministic"})


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolSpec:
    name: str
    intent: Intent
    description: str
    args_model: type[ToolArgs]
    needs_llm: bool
    stage: Stage | None                       # artifact this tool produces
    prerequisites: tuple[Stage, ...]          # stages that must be FRESH
    build_kwargs: Callable[[Project, ToolArgs, LLMProvider | None], dict[str, Any]]
    func: Callable[..., ToolResult]
    apply: Callable[[Project, ToolResult], None]


def _generation_kwargs(project: Project) -> dict[str, Any]:
    return dict(duration_minutes=project.duration_minutes, constraints=project.constraints,
                language=project.language)


def _kw_develop_idea(p: Project, a: ToolArgs, llm) -> dict:
    return dict(llm=llm, idea=p.idea, genre=p.genre, approved_decisions=p.approved_decisions,
                instructions=a.instructions, **_generation_kwargs(p))


def _kw_build_script(p: Project, a: ToolArgs, llm) -> dict:
    return dict(llm=llm, idea=p.idea, genre=p.genre, approved_decisions=p.approved_decisions,
                visual_style=p.visual_style, max_locations=p.max_locations, max_actors=p.max_actors,
                instructions=a.instructions, previous_logline=p.logline, previous_synopsis=p.synopsis,
                **_generation_kwargs(p))


def _kw_breakdown(p: Project, a: ToolArgs, llm) -> dict:
    return dict(llm=llm, logline=p.logline, characters=p.characters, script_scenes=p.script_scenes,
                max_locations=p.max_locations, max_actors=p.max_actors, instructions=a.instructions,
                **_generation_kwargs(p))


def _kw_shots(p: Project, a: ToolArgs, llm) -> dict:
    return dict(llm=llm, scenes=p.scenes, characters=p.characters, visual_style=p.visual_style,
                equipment=p.equipment, constraints=p.constraints, language=p.language,
                instructions=a.instructions)


def _kw_check(p: Project, a: ToolArgs, llm) -> dict:
    return dict(characters=p.characters, scenes=p.scenes,
                shots=p.shots if p.is_fresh(Stage.SHOTS) else [],
                duration_minutes=p.duration_minutes, max_locations=p.max_locations,
                max_actors=p.max_actors, equipment=p.equipment, llm=llm, use_model=a.use_model,
                language=p.language)


def _kw_plan(p: Project, a: ToolArgs, llm) -> dict:
    return dict(scenes=p.scenes, shots=p.shots, equipment=p.equipment, budget=p.budget.value,
                max_actors=p.max_actors, language=p.language)


def _kw_export(p: Project, a: ToolArgs, llm) -> dict:
    return dict(project=p)


TOOL_REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec("develop_idea", Intent.DEVELOP_IDEA,
                 "Develop the film idea: logline options, themes, suggested decisions, questions.",
                 InstructionArgs, True, Stage.IDEA, (), _kw_develop_idea, develop_idea,
                 lambda p, r: p.apply_idea(IdeaNotes.model_validate(r.data), r.meta.get("origin", "model"))),
        ToolSpec("build_script", Intent.BUILD_SCRIPT,
                 "Write logline, synopsis, characters and screenplay scenes from the idea.",
                 InstructionArgs, True, Stage.SCRIPT, (), _kw_build_script, build_script,
                 lambda p, r: p.apply_script(ScriptOutput.model_validate(r.data), r.meta.get("origin", "model"))),
        ToolSpec("breakdown_scenes", Intent.BREAKDOWN_SCENES,
                 "Break the script into production scenes (location, time, characters, duration).",
                 InstructionArgs, True, Stage.SCENES, (Stage.SCRIPT,), _kw_breakdown, breakdown_scenes,
                 lambda p, r: p.apply_scenes(SceneBreakdownOutput.model_validate(r.data).scenes,
                                             r.meta.get("origin", "model"))),
        ToolSpec("create_shot_list", Intent.CREATE_SHOT_LIST,
                 "Create a numbered shot list linked to the scenes.",
                 InstructionArgs, True, Stage.SHOTS, (Stage.SCENES,), _kw_shots, create_shot_list,
                 lambda p, r: p.apply_shots(ShotListOutput.model_validate(r.data).shots,
                                            r.meta.get("origin", "model"))),
        ToolSpec("estimate_resources", Intent.PLAN_PRODUCTION,
                 "Suggest props, equipment, locations and a filming schedule (rule-based estimates).",
                 NoArgs, False, Stage.PLAN, (Stage.SCENES, Stage.SHOTS), _kw_plan, estimate_resources,
                 lambda p, r: p.apply_plan(ProductionPlan.model_validate(r.data), "deterministic")),
        ToolSpec("check_continuity", Intent.CHECK_CONTINUITY,
                 "Validate characters, scene/shot references, durations and constraints.",
                 CheckArgs, False, Stage.VALIDATION, (Stage.SCENES,), _kw_check, check_continuity,
                 lambda p, r: p.apply_validation(r.issues, r.meta.get("origin", "deterministic"))),
        ToolSpec("export_package", Intent.EXPORT_PACKAGE,
                 "Export the project as a Markdown production document.",
                 NoArgs, False, None, (), _kw_export, export_package, lambda p, r: None),
    ]
}

INTENT_TO_TOOL: dict[Intent, str] = {spec.intent: spec.name for spec in TOOL_REGISTRY.values()}


def describe_tools() -> str:
    """Tool list for the routing prompt."""
    lines = []
    for spec in TOOL_REGISTRY.values():
        args = ", ".join(spec.args_model.model_fields) or "none"
        lines.append(f"- {spec.name} (intent {spec.intent.value}; arguments: {args}): {spec.description}")
    return "\n".join(lines)
