"""Orchestrator tests with a scripted (test-only) LLM. No network, no Ollama."""
from __future__ import annotations

from conftest import ScriptedLLM, scenes_reply, script_reply, shots_reply

from agent import MAX_TOOL_CALLS, FrameAgent, detect_intent, is_pipeline_request, next_intent
from llm_provider import LLMConnectionError
from sample_project import sample_brief, sample_full
from schemas import ArtifactStatus, Intent, Stage
from tools import TOOL_REGISTRY, ToolSpec


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_keyword_routing_mongolian_and_english():
    assert detect_intent("Сценар бич") == Intent.BUILD_SCRIPT
    assert detect_intent("Дүр зургийг задлаарай") == Intent.BREAKDOWN_SCENES
    assert detect_intent("Shot list үүсгэ") == Intent.CREATE_SHOT_LIST
    assert detect_intent("Please validate continuity") == Intent.CHECK_CONTINUITY
    assert detect_intent("Production schedule гарга") == Intent.PLAN_PRODUCTION
    assert detect_intent("Export the package") == Intent.EXPORT_PACKAGE
    assert detect_intent("Санаагаа хөгжүүл") == Intent.DEVELOP_IDEA
    assert detect_intent("Why do we need a shot list?") is None      # a question, not a command
    assert detect_intent("hello there") is None
    assert is_pipeline_request("Create the full script package")


def test_next_intent_follows_the_workflow():
    p = sample_brief()
    assert next_intent(p) == Intent.BUILD_SCRIPT
    p.idea = ""
    assert next_intent(p) == Intent.DEVELOP_IDEA
    assert next_intent(sample_full()) == Intent.CHECK_CONTINUITY


def test_unknown_tool_proposed_by_model_is_rejected(full_project):
    llm = ScriptedLLM([{"intent": "BUILD_SCRIPT", "tool": "delete_everything", "arguments": {}}])
    before = full_project.model_dump_json()
    turn = FrameAgent(llm).handle(full_project, "please look this over for me")
    assert not turn.ok and "delete_everything" in turn.reply
    assert turn.steps == []
    assert full_project.model_dump_json() == before                  # nothing executed, nothing changed


def test_invalid_tool_arguments_from_model_are_rejected(full_project):
    llm = ScriptedLLM([{"intent": "CHECK_CONTINUITY", "tool": "check_continuity",
                        "arguments": {"shell": "rm -rf /"}}])
    turn = FrameAgent(llm).handle(full_project, "please look this over for me")
    assert not turn.ok and turn.steps == []


def test_valid_model_proposal_is_executed(full_project):
    llm = ScriptedLLM([{"intent": "CHECK_CONTINUITY", "tool": "check_continuity", "arguments": {}}])
    turn = FrameAgent(llm).handle(full_project, "please look this over for me")
    assert turn.ok and [s.tool for s in turn.steps] == ["check_continuity"]
    assert full_project.is_fresh(Stage.VALIDATION)


def test_invalid_router_json_falls_back_to_general_answer(full_project):
    llm = ScriptedLLM(["garbage", "garbage again", "Шот гэдэг нь тасралтгүй авсан кадр юм."])
    turn = FrameAgent(llm).handle(full_project, "please look this over for me")
    assert turn.intent == Intent.GENERAL_QUESTION and turn.steps == []
    assert "кадр" in turn.reply and "Дараагийн алхам" in turn.reply


# --------------------------------------------------------------------------- #
# Clarification, prerequisites, availability
# --------------------------------------------------------------------------- #
def test_missing_premise_triggers_clarifying_question():
    p = sample_brief()
    p.idea = ""
    llm = ScriptedLLM()
    turn = FrameAgent(llm).handle(p, "", forced_intent=Intent.BUILD_SCRIPT)
    assert turn.needs_clarification and "санаа" in turn.reply.lower()
    assert llm.calls == [] and p.get_status(Stage.SCRIPT) == ArtifactStatus.EMPTY


def test_missing_genre_is_asked():
    p = sample_brief()
    p.genre = ""
    turn = FrameAgent(ScriptedLLM()).handle(p, "", forced_intent=Intent.BUILD_SCRIPT)
    assert turn.needs_clarification and "жанр" in turn.reply


def test_prerequisite_blocks_execution_and_suggests_producer(brief_project):
    llm = ScriptedLLM()
    turn = FrameAgent(llm).handle(brief_project, "", forced_intent=Intent.CREATE_SHOT_LIST)
    assert llm.calls == [] and turn.steps == []
    assert turn.suggested_intent == Intent.BREAKDOWN_SCENES or turn.suggested_intent == Intent.BUILD_SCRIPT


def test_outdated_dependency_is_not_used(full_project):
    full_project.status[Stage.SCENES] = ArtifactStatus.OUTDATED
    turn = FrameAgent(ScriptedLLM()).handle(full_project, "", forced_intent=Intent.CREATE_SHOT_LIST)
    assert turn.steps == [] and "хуучирсан" in turn.reply
    assert turn.suggested_intent == Intent.BREAKDOWN_SCENES


def test_unavailable_model_gives_clear_error_and_keeps_state(brief_project):
    llm = ScriptedLLM(available=False)
    before = brief_project.model_dump_json()
    turn = FrameAgent(llm).handle(brief_project, "", forced_intent=Intent.BUILD_SCRIPT)
    assert not turn.ok and "unavailable" in turn.reply and turn.steps == []
    assert brief_project.model_dump_json() == before


def test_deterministic_tools_work_without_any_llm(full_project):
    agent = FrameAgent(None)
    turn = agent.handle(full_project, "validate continuity")
    assert turn.ok and full_project.is_fresh(Stage.VALIDATION)
    turn = agent.handle(full_project, "export the package")
    assert turn.ok and turn.export_markdown and "# The Last Bus" in turn.export_markdown
    turn = agent.handle(full_project, "", forced_intent=Intent.BUILD_SCRIPT)
    assert not turn.ok and "загвар" in turn.reply


def test_connection_error_during_tool_is_reported_without_state_change(brief_project):
    llm = ScriptedLLM([LLMConnectionError("Ollama руу холбогдож чадсангүй")])
    before = brief_project.model_dump_json()
    turn = FrameAgent(llm).handle(brief_project, "", forced_intent=Intent.BUILD_SCRIPT)
    assert not turn.ok and "холбогдож чадсангүй" in turn.reply
    assert brief_project.model_dump_json() == before


def test_invalid_model_output_never_updates_project(brief_project):
    llm = ScriptedLLM(["not json", "{ nope"])
    turn = FrameAgent(llm).handle(brief_project, "", forced_intent=Intent.BUILD_SCRIPT)
    assert not turn.ok and "JSON" in turn.reply
    assert brief_project.script_scenes == [] and brief_project.logline == ""
    assert len(llm.calls) == 2


def test_tool_crash_is_captured(full_project):
    def exploding(**kwargs):
        raise RuntimeError("kaboom")

    spec = TOOL_REGISTRY["check_continuity"]
    broken = ToolSpec(spec.name, spec.intent, spec.description, spec.args_model, spec.needs_llm, spec.stage,
                      spec.prerequisites, spec.build_kwargs, exploding, spec.apply)
    agent = FrameAgent(None, registry={**TOOL_REGISTRY, "check_continuity": broken})
    turn = agent.handle(full_project, "", forced_intent=Intent.CHECK_CONTINUITY)
    assert not turn.ok and "kaboom" in turn.reply
    assert not full_project.is_fresh(Stage.VALIDATION)


# --------------------------------------------------------------------------- #
# State invalidation and creative-decision preservation
# --------------------------------------------------------------------------- #
def test_regenerating_script_marks_dependents_outdated(full_project):
    agent = FrameAgent(ScriptedLLM([script_reply()]))
    agent.handle(full_project, "", forced_intent=Intent.CHECK_CONTINUITY)
    assert full_project.is_fresh(Stage.VALIDATION)

    turn = agent.handle(full_project, "", forced_intent=Intent.BUILD_SCRIPT)
    assert turn.ok
    assert full_project.is_fresh(Stage.SCRIPT)
    for stage in (Stage.SCENES, Stage.SHOTS, Stage.PLAN, Stage.VALIDATION):
        assert full_project.get_status(stage) == ArtifactStatus.OUTDATED
    assert full_project.scenes and full_project.shots                 # kept visible, but flagged
    assert "Хуучирсан" in turn.reply
    assert full_project.provenance[Stage.SCRIPT] == "model:scripted-test-model"


def test_regenerating_scenes_only_outdates_later_stages(full_project):
    turn = FrameAgent(ScriptedLLM([scenes_reply()])).handle(full_project, "", forced_intent=Intent.BREAKDOWN_SCENES)
    assert turn.ok
    assert full_project.is_fresh(Stage.SCRIPT) and full_project.is_fresh(Stage.SCENES)
    assert full_project.get_status(Stage.SHOTS) == ArtifactStatus.OUTDATED
    assert full_project.get_status(Stage.PLAN) == ArtifactStatus.OUTDATED


def test_approved_decisions_are_sent_to_the_model_and_preserved(full_project):
    decision = full_project.approved_decisions[0]
    llm = ScriptedLLM([script_reply()])
    FrameAgent(llm).handle(full_project, "", forced_intent=Intent.BUILD_SCRIPT,
                           forced_args={"instructions": "Төгсгөлийг илүү түгшүүртэй болго"})
    prompt = llm.calls[0][-1]["content"]
    assert decision in prompt and "MUST be preserved" in prompt
    assert "Төгсгөлийг илүү түгшүүртэй болго" in prompt
    assert full_project.approved_decisions == [decision]


def test_change_request_in_chat_becomes_instructions(full_project):
    llm = ScriptedLLM([script_reply()])
    FrameAgent(llm).handle(full_project, "Сценарыг дахин бич, төгсгөлийг өөрчил")
    assert "төгсгөлийг өөрчил" in llm.calls[0][-1]["content"]


# --------------------------------------------------------------------------- #
# Bounded workflow and complete flow
# --------------------------------------------------------------------------- #
def test_at_most_four_tools_per_turn(brief_project):
    llm = ScriptedLLM([script_reply(), scenes_reply(), shots_reply()])
    turn = FrameAgent(llm).handle(brief_project, "Create the full script package")
    assert MAX_TOOL_CALLS == 4
    assert [s.tool for s in turn.steps] == ["build_script", "breakdown_scenes", "create_shot_list",
                                            "estimate_resources"]
    assert brief_project.get_status(Stage.VALIDATION) == ArtifactStatus.EMPTY   # 5th step not run
    assert "дээд тал нь 4" in turn.reply
    assert turn.ok


def test_custom_tool_budget_is_enforced(brief_project):
    llm = ScriptedLLM([script_reply(), scenes_reply(), shots_reply()])
    turn = FrameAgent(llm, max_tool_calls=2).handle(brief_project, "Create the full script package")
    assert len(turn.steps) == 2 and len(llm.calls) == 2


def test_pipeline_stops_at_first_failure(brief_project):
    llm = ScriptedLLM([script_reply(), "bad", "still bad"])
    turn = FrameAgent(llm).handle(brief_project, "Create the full script package")
    assert [s.ok for s in turn.steps] == [True, False]
    assert not turn.ok and brief_project.is_fresh(Stage.SCRIPT)
    assert brief_project.get_status(Stage.SCENES) == ArtifactStatus.EMPTY


def test_complete_workflow_with_mocked_llm(brief_project):
    idea_reply = {"summary": "Сайн санаа", "logline_options": ["Логлайн A", "Логлайн B"],
                  "themes": ["ирээдүй"], "suggested_decisions": ["Эр хамгаалагч"],
                  "story_questions": ["Төгсгөл хэн?"]}
    llm = ScriptedLLM([idea_reply, script_reply(), scenes_reply(), shots_reply()])
    agent = FrameAgent(llm)
    statuses: list[str] = []

    # 1 idea -> 2 script/scenes/shots/plan in one bounded turn -> 3 check -> 4 export
    t1 = agent.handle(brief_project, "Санаагаа хөгжүүл", on_status=statuses.append)
    assert t1.ok and brief_project.idea_notes.logline_options == ["Логлайн A", "Логлайн B"]
    assert t1.suggested_intent == Intent.BUILD_SCRIPT

    t2 = agent.handle(brief_project, "Build the full script package", on_status=statuses.append)
    assert t2.ok and len(t2.steps) == 4
    assert len(brief_project.shots) == 15 and brief_project.plan.shoot_days >= 1

    t3 = agent.handle(brief_project, "Check continuity")
    assert t3.ok and brief_project.is_fresh(Stage.VALIDATION)
    assert brief_project.issues == []                                   # the data respects its constraints
    assert t3.suggested_intent == Intent.EXPORT_PACKAGE

    t4 = agent.handle(brief_project, "Export the package")
    assert t4.ok and "# The Last Bus" in t4.export_markdown and t4.export_filename.endswith(".md")
    assert "AI-generated" in t4.export_markdown and "AI-generated · scripted-test-model" in t4.export_markdown

    assert "Төслийг шинжилж байна" in statuses and any("Сценар" in s for s in statuses)
    assert llm.replies == []                                            # every scripted reply was consumed
    assert all(brief_project.is_fresh(s) for s in Stage)


def test_first_message_becomes_the_premise():
    from schemas import Project
    p = Project()
    llm = ScriptedLLM([{"summary": "s", "logline_options": ["L"], "themes": [], "suggested_decisions": [],
                        "story_questions": []}])
    msg = "Хөгшин загасчин далайгаас нууц самбар олдог тухай богино кино хийе гэж бодож байна"
    turn = FrameAgent(llm).handle(p, msg)
    assert turn.ok and p.idea == msg and p.is_fresh(Stage.IDEA)


def test_custom_registry_only_allows_registered_tools(full_project):
    agent = FrameAgent(None, registry={"check_continuity": TOOL_REGISTRY["check_continuity"]})
    turn = agent.handle(full_project, "", forced_intent=Intent.EXPORT_PACKAGE)
    assert not turn.ok and "export_package" in turn.reply
