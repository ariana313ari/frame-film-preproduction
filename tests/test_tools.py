"""Tool-level tests. No network, no Ollama."""
from __future__ import annotations

import pytest
import requests
from conftest import ScriptedLLM, scenes_reply, script_reply, shots_reply

import tools
from export_utils import project_to_markdown
from llm_provider import (
    InvalidModelOutputError,
    LLMConfig,
    LLMConnectionError,
    LLMModelNotFoundError,
    OllamaProvider,
    extract_json,
    generate_validated,
)
from sample_project import sample_brief, sample_full, sample_with_flaws
from schemas import (
    ArtifactStatus,
    CameraMovement,
    Scene,
    SceneBreakdownOutput,
    Severity,
    Shot,
    ShotType,
    Stage,
    TimeOfDay,
)


def check(project, **overrides):
    kwargs = dict(characters=project.characters, scenes=project.scenes, shots=project.shots,
                  duration_minutes=project.duration_minutes, max_locations=project.max_locations,
                  max_actors=project.max_actors, equipment=project.equipment)
    kwargs.update(overrides)
    return tools.check_continuity(**kwargs)


def codes(result):
    return {i.code for i in result.issues}


# --------------------------------------------------------------------------- #
# breakdown_scenes
# --------------------------------------------------------------------------- #
def test_valid_scene_breakdown(full_project):
    llm = ScriptedLLM([scenes_reply()])
    result = tools.breakdown_scenes(
        llm, logline=full_project.logline, characters=full_project.characters,
        script_scenes=full_project.script_scenes, duration_minutes=3, max_locations=1, max_actors=2)
    assert result.ok, result.error
    scenes = SceneBreakdownOutput.model_validate(result.data).scenes
    assert [s.number for s in scenes] == [1, 2, 3, 4]
    assert all(s.location and s.estimated_duration_sec > 0 for s in scenes)
    assert scenes[0].time_of_day == TimeOfDay.NIGHT
    assert result.warnings == []  # the sample respects its own constraints


def test_breakdown_requires_a_script():
    result = tools.breakdown_scenes(ScriptedLLM(), logline="", characters=[], script_scenes=[],
                                    duration_minutes=3)
    assert not result.ok and "Сценар" in result.error


def test_breakdown_repairs_non_positive_duration_once(full_project):
    bad = scenes_reply()
    bad["scenes"][0]["estimated_duration_sec"] = 0
    llm = ScriptedLLM([bad, scenes_reply()])
    result = tools.breakdown_scenes(
        llm, logline="x", characters=full_project.characters,
        script_scenes=full_project.script_scenes, duration_minutes=3)
    assert result.ok and len(llm.calls) == 2
    assert "positive" in llm.calls[1][-1]["content"]  # the repair prompt explains the problem


# --------------------------------------------------------------------------- #
# create_shot_list: numbering, durations, scene links
# --------------------------------------------------------------------------- #
def test_shot_list_is_renumbered_and_linked(full_project):
    reply = shots_reply()
    reply["shots"].reverse()                      # out of order, wrong numbers kept
    llm = ScriptedLLM([reply])
    result = tools.create_shot_list(llm, scenes=full_project.scenes, characters=full_project.characters)
    assert result.ok, result.error
    shots = [Shot.model_validate(s) for s in result.data["shots"]]
    assert [s.number for s in shots] == list(range(1, len(shots) + 1))
    assert [s.scene_number for s in shots] == sorted(s.scene_number for s in shots)
    assert all(s.estimated_duration_sec > 0 for s in shots)


def test_shot_list_rejects_unknown_scene_reference_after_one_repair(full_project):
    bad = shots_reply()
    bad["shots"][0]["scene_number"] = 9
    llm = ScriptedLLM([bad, bad])
    result = tools.create_shot_list(llm, scenes=full_project.scenes, characters=full_project.characters)
    assert not result.ok
    assert len(llm.calls) == 2                    # exactly one repair attempt
    assert "Valid scene numbers" in llm.calls[1][-1]["content"]


def test_shot_list_rejects_non_positive_duration(full_project):
    bad = shots_reply()
    bad["shots"][2]["estimated_duration_sec"] = -3
    result = tools.create_shot_list(ScriptedLLM([bad, bad]), scenes=full_project.scenes,
                                    characters=full_project.characters)
    assert not result.ok


# --------------------------------------------------------------------------- #
# check_continuity (deterministic rules)
# --------------------------------------------------------------------------- #
def test_clean_sample_has_no_findings(full_project):
    result = check(full_project)
    assert result.ok and result.issues == []
    assert result.meta["origin"] == "deterministic"


def test_invalid_scene_reference_is_an_error(full_project):
    full_project.shots[3].scene_number = 42
    result = check(full_project)
    issue = next(i for i in result.issues if i.code == "SHOT_SCENE_MISSING")
    assert issue.severity == Severity.ERROR and issue.shot_number == 4 and issue.suggestion


def test_unknown_character_is_reported(full_project):
    full_project.scenes[1].characters.append("Жолооч")
    result = check(full_project)
    issue = next(i for i in result.issues if i.code == "CHARACTER_UNKNOWN")
    assert "Жолооч" in issue.message and issue.scene_number == 2


def test_character_matching_ignores_case_and_spacing(full_project):
    full_project.scenes[0].characters = ["  номин ", "ТАНИХГҮЙ  ЭР"]
    assert "CHARACTER_UNKNOWN" not in codes(check(full_project))


def test_unused_character_is_info(full_project):
    for s in full_project.scenes:
        s.characters = ["Номин"]
    result = check(full_project)
    unused = next(i for i in result.issues if i.code == "CHARACTER_UNUSED")
    assert unused.severity == Severity.INFO


def test_scene_duration_must_be_positive(full_project):
    full_project.scenes[0].estimated_duration_sec = 0
    result = check(full_project)
    assert any(i.code == "SCENE_DURATION_INVALID" and i.severity == Severity.ERROR for i in result.issues)


@pytest.mark.parametrize("scale, expected", [(1.0, None), (1.25, Severity.WARNING), (2.0, Severity.ERROR),
                                             (0.5, Severity.ERROR)])
def test_project_duration_mismatch_levels(full_project, scale, expected):
    for s in full_project.scenes:
        s.estimated_duration_sec *= scale
    found = [i for i in check(full_project, shots=[]).issues if i.code == "FILM_DURATION_MISMATCH"]
    if expected is None:
        assert not found
    else:
        assert found and found[0].severity == expected


def test_location_and_actor_limits(full_project):
    full_project.scenes[2].location = "Дэлгүүр"
    full_project.scenes[2].characters = ["Номин", "Танихгүй эр", "Жолооч"]
    result = check(full_project)
    assert {"LOCATION_LIMIT", "ACTOR_LIMIT", "ACTOR_LIMIT_SCENE"} <= codes(result)


def test_duplicate_and_gapped_shot_numbers(full_project):
    full_project.shots[1].number = 1
    full_project.shots[5].number = 99
    result = check(full_project)
    assert {"SHOT_NUMBER_DUPLICATE", "SHOT_NUMBER_GAP"} <= codes(result)


def test_scene_without_shots_and_shot_duration_mismatch(full_project):
    full_project.shots = [s for s in full_project.shots if s.scene_number != 4]
    assert "SCENE_WITHOUT_SHOTS" in codes(check(full_project))
    full_project.shots[0].estimated_duration_sec = 200
    assert "SHOT_SCENE_DURATION_MISMATCH" in codes(check(full_project))


def test_equipment_check_only_when_equipment_is_listed(full_project):
    full_project.shots[0].movement = CameraMovement.DOLLY
    assert "EQUIPMENT_MISSING" in codes(check(full_project))
    assert "EQUIPMENT_MISSING" not in codes(check(full_project, equipment=[]))


def test_prop_heuristics_are_marked_as_info(full_project):
    full_project.shots[0].description += " Ширээн дээр зураг байна."
    issue = next(i for i in check(full_project).issues if i.code == "PROP_NOT_LISTED")
    assert issue.severity == Severity.INFO and "таамаглал" in issue.message


def test_check_continuity_needs_scenes():
    result = tools.check_continuity(characters=[], scenes=[], shots=[], duration_minutes=3)
    assert not result.ok and result.error


def test_model_review_is_labelled_and_failure_does_not_hide_rule_results(full_project):
    llm = ScriptedLLM([{"issues": [{"severity": "ERROR", "message": "Шүхэр өнгө өөрчлөгдсөн",
                                    "suggestion": "Нэг өнгөөр хадгал", "scene_number": 3}]}])
    result = check(full_project, llm=llm, use_model=True)
    model_issues = [i for i in result.issues if i.source == "model"]
    assert len(model_issues) == 1
    assert model_issues[0].severity == Severity.WARNING          # model output is never promoted to ERROR
    assert result.meta["origin"].startswith("deterministic+model")

    flawed = sample_with_flaws()
    failing = ScriptedLLM([LLMConnectionError("down")])
    result = check(flawed, llm=failing, use_model=True)
    assert "SHOT_SCENE_MISSING" in codes(result)                  # deterministic findings survive
    assert "MODEL_REVIEW_FAILED" in codes(result) and result.warnings


# --------------------------------------------------------------------------- #
# estimate_resources
# --------------------------------------------------------------------------- #
def test_resource_estimate_has_assumptions_and_no_prices(full_project):
    result = tools.estimate_resources(scenes=full_project.scenes, shots=full_project.shots,
                                      equipment=full_project.equipment, budget="LOW", max_actors=2)
    assert result.ok, result.error
    plan = result.data
    assert plan["schedule"] and plan["assumptions"]
    assert any("ТААМАГЛАЛ" in a for a in plan["assumptions"])
    assert {p["name"] for p in plan["props"]} >= {"улаан шүхэр", "зураг"}
    assert all("price" not in str(v).lower() and "₮" not in str(v) for v in plan.values())
    assert sorted(b["day"] for b in plan["schedule"])[0] == 1


def test_resource_estimate_requires_shots(full_project):
    result = tools.estimate_resources(scenes=full_project.scenes, shots=[])
    assert not result.ok and "Шотын" in result.error


# --------------------------------------------------------------------------- #
# Invalid model output / LLM failures
# --------------------------------------------------------------------------- #
def test_invalid_model_output_gives_error_not_fabricated_data(brief_project):
    llm = ScriptedLLM(["this is not json", "{still: not json"])
    result = tools.build_script(llm, idea=brief_project.idea, genre="thriller", duration_minutes=3)
    assert not result.ok and "JSON" in result.error
    assert result.data == {}
    assert len(llm.calls) == 2                                   # original + one repair


def test_schema_invalid_output_is_repaired_once(brief_project):
    good = script_reply()
    llm = ScriptedLLM([{"logline": "x"}, good])
    result = tools.build_script(llm, idea=brief_project.idea, genre="thriller", duration_minutes=3)
    assert result.ok
    assert [s["number"] for s in result.data["scenes"]] == [1, 2, 3, 4]


def test_connection_failure_becomes_a_tool_error(brief_project):
    llm = ScriptedLLM([LLMConnectionError("Ollama руу холбогдож чадсангүй")])
    result = tools.build_script(llm, idea=brief_project.idea, genre="thriller", duration_minutes=3)
    assert not result.ok and "холбогдож" in result.error


def test_build_script_validates_inputs():
    assert not tools.build_script(ScriptedLLM(), idea="", genre="x", duration_minutes=3).ok
    assert not tools.build_script(ScriptedLLM(), idea="x", genre="", duration_minutes=3).ok
    assert not tools.build_script(ScriptedLLM(), idea="x", genre="y", duration_minutes=0).ok
    assert not tools.build_script(None, idea="x", genre="y", duration_minutes=3).ok


def test_ollama_connection_refused_is_reported_clearly():
    provider = OllamaProvider(LLMConfig(base_url="http://127.0.0.1:9", timeout_seconds=2))
    status = provider.check_status()
    assert not status.available and "Ollama" in status.message
    with pytest.raises(LLMConnectionError):
        provider.chat([{"role": "user", "content": "hi"}])


def test_ollama_missing_model(monkeypatch):
    provider = OllamaProvider(LLMConfig(model="nope:1b"))

    class Resp:
        status_code = 404
        text = '{"error":"model not found"}'

        def json(self):
            return {"error": "model 'nope:1b' not found"}

    monkeypatch.setattr(provider._session, "post", lambda *a, **k: Resp())
    with pytest.raises(LLMModelNotFoundError, match="ollama pull"):
        provider.chat([{"role": "user", "content": "hi"}])

    class Tags:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "qwen2.5:3b"}]}

    monkeypatch.setattr(provider._session, "get", lambda *a, **k: Tags())
    assert not provider.check_status().available
    ok = OllamaProvider(LLMConfig(model="qwen2.5:3b"))
    monkeypatch.setattr(ok._session, "get", lambda *a, **k: Tags())
    assert ok.check_status().available


def test_ollama_timeout(monkeypatch):
    provider = OllamaProvider()

    def boom(*a, **k):
        raise requests.Timeout()

    monkeypatch.setattr(provider._session, "post", boom)
    from llm_provider import LLMTimeoutError
    with pytest.raises(LLMTimeoutError):
        provider.chat([{"role": "user", "content": "hi"}])


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_generate_validated_raises_after_repair_budget():
    llm = ScriptedLLM(["nope", "still nope"])
    with pytest.raises(InvalidModelOutputError) as err:
        generate_validated(llm, [{"role": "user", "content": "x"}], SceneBreakdownOutput)
    assert err.value.raw == "still nope"


def test_enum_values_are_normalised():
    shot = Shot.model_validate({"number": 1, "scene_number": 1, "shot_type": "close-up",
                                "movement": "hand held", "camera_angle": "low angle"})
    assert shot.shot_type == ShotType.CLOSE_UP and shot.movement == CameraMovement.HANDHELD
    scene = Scene.model_validate({"number": 1, "location": "x", "time_of_day": "ШӨНӨ",
                                  "characters": "A, B"})
    assert scene.time_of_day == TimeOfDay.NIGHT and scene.characters == ["A", "B"]
    with pytest.raises(ValueError):
        Shot.model_validate({"number": 1, "scene_number": 1, "shot_type": "teleport"})


# --------------------------------------------------------------------------- #
# export_package
# --------------------------------------------------------------------------- #
def test_markdown_export_contains_all_sections(full_project):
    result = tools.export_package(project=full_project)
    assert result.ok, result.error
    md = result.data["markdown"]
    for heading in ("# The Last Bus", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.",
                    "## 8.", "## 9.", "## 10."):
        assert heading in md
    assert "SAMPLE" in md or "Sample data" in md              # static data is labelled as such
    assert "Номин" in md and "| # | Scene | Type |" in md
    assert "Rule-based checks" in md and "AI-generated suggestions" in md
    assert result.data["filename"].endswith(".md")
    # No validation was run -> export attaches fresh deterministic findings and says so
    assert any("Шалгалт хийгдээгүй" in w for w in result.warnings)


def test_export_marks_outdated_sections_and_escapes_pipes(full_project):
    full_project.status[Stage.SHOTS] = ArtifactStatus.OUTDATED
    full_project.scenes[0].action = "a | b"
    md = project_to_markdown(full_project)
    assert "Outdated sections" in md and "a \\| b" in md


def test_export_refuses_empty_project():
    result = tools.export_package(project=sample_brief())
    assert not result.ok and "агуулга" in result.error


def test_plan_in_export_lists_assumptions(full_project):
    md = tools.export_package(project=full_project).data["markdown"]
    assert "Assumptions" in md and "ТААМАГЛАЛ" in md
