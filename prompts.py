"""All model prompts live here.

Instructions are written in English (small local models follow them more reliably);
the language of the *generated content* is controlled by `language_rule`.
Every prompt that expects JSON shows the exact shape and enum values allowed.
"""
from __future__ import annotations

import json
from typing import Iterable

from schemas import (
    CameraAngle,
    CameraMovement,
    Character,
    Scene,
    ScriptScene,
    ShotType,
    TimeOfDay,
)

SYSTEM = (
    "You are FRAME, a careful film pre-production assistant for beginner filmmakers. "
    "When JSON is requested, reply with ONE valid JSON object only: no markdown, no comments, "
    "no text before or after it. Never invent facts about the real world, prices or research."
)


def language_rule(language: str) -> str:
    if language == "en":
        return "Write all human-readable string values in English."
    return (
        "Write all human-readable string values in Mongolian (Cyrillic). "
        "Keep JSON keys and enum values (e.g. shot_type, camera_angle, movement, time_of_day) "
        "in English exactly as specified. Technical film terms such as INT./EXT. may stay in English."
    )


def _enum_values(enum_cls) -> str:
    return " | ".join(m.value for m in enum_cls)


def repair_prompt(problem: str) -> str:
    return (
        "Your previous answer was rejected by the validator.\n"
        f"Problems: {problem}\n"
        "Return the complete corrected answer as ONE valid JSON object, following the required "
        "structure exactly. Do not add any text outside the JSON."
    )


def target_scene_range(duration_minutes: float) -> tuple[int, int]:
    low = max(2, round(duration_minutes * 1.0))
    high = min(10, max(low + 1, round(duration_minutes * 1.7)))
    return low, high


def _limits_text(max_locations: int | None, max_actors: int | None) -> str:
    lines = []
    if max_locations:
        lines.append(f"- Use at most {max_locations} distinct location(s).")
    if max_actors:
        lines.append(f"- Use at most {max_actors} on-screen character(s)/actor(s).")
    return "\n".join(lines) or "- No hard location/actor limits."


def _msgs(user: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Tool prompts
# --------------------------------------------------------------------------- #
def develop_idea_messages(*, idea, genre, duration_minutes, constraints, approved_decisions,
                          language, instructions="") -> list[dict]:
    approved = "\n".join(f"- {d}" for d in approved_decisions) or "- (none yet)"
    return _msgs(f"""Help a beginner filmmaker develop a short-film idea.
{language_rule(language)}

Film idea: {idea}
Genre: {genre or 'not decided'}
Target duration: {duration_minutes:g} minutes
Constraints: {constraints or 'none'}
Decisions already approved by the user (keep them, never contradict them):
{approved}
{('Extra instruction from the user: ' + instructions) if instructions else ''}

Return JSON with exactly this shape:
{{
  "summary": "2-3 sentences reacting to the idea and what makes it filmable",
  "logline_options": ["2 or 3 alternative one-sentence loglines"],
  "themes": ["2-4 short themes"],
  "suggested_decisions": ["3-5 concrete creative decisions the user could approve, e.g. who the secret belongs to"],
  "story_questions": ["2-4 questions that would help to make the story clearer"]
}}""")


def build_script_messages(*, idea, genre, duration_minutes, constraints, approved_decisions,
                          visual_style, max_locations, max_actors, language,
                          instructions="", previous_logline="", previous_synopsis="") -> list[dict]:
    low, high = target_scene_range(duration_minutes)
    approved = "\n".join(f"- {d}" for d in approved_decisions) or "- (none)"
    previous = ""
    if instructions and previous_logline:
        previous = (f"\nCurrent version (change only what the instruction asks for):\n"
                    f"Logline: {previous_logline}\nSynopsis: {previous_synopsis}\n")
    return _msgs(f"""Write a short-film script package.
{language_rule(language)}

Premise: {idea}
Genre: {genre}
Target duration: {duration_minutes:g} minutes (about {duration_minutes * 60:.0f} seconds of screen time)
Visual style: {visual_style or 'not specified'}
Constraints: {constraints or 'none'}
Hard limits:
{_limits_text(max_locations, max_actors)}
Creative decisions approved by the user - they MUST be preserved:
{approved}
{('User instruction for this version: ' + instructions) if instructions else ''}{previous}
Rules:
- Write {low}-{high} scenes, short enough to film within the target duration.
- Each scene "text" is screenplay style: action lines plus short dialogue (CHARACTER NAME in capitals before each line).
- Reuse the exact same character names everywhere.
- "details" lists fixed visual details that must stay consistent (costume, key props).

Return JSON with exactly this shape:
{{
  "logline": "one sentence",
  "synopsis": "3-6 sentences",
  "characters": [{{"name": "...", "role": "protagonist | antagonist | supporting", "description": "...", "details": ["red raincoat"]}}],
  "scenes": [{{"number": 1, "heading": "INT./EXT. LOCATION - TIME", "text": "action and dialogue"}}]
}}""")


def breakdown_scenes_messages(*, logline, characters: Iterable[Character],
                              script_scenes: Iterable[ScriptScene], duration_minutes,
                              max_locations, max_actors, constraints, language,
                              instructions="") -> list[dict]:
    names = [c.name for c in characters]
    script = [s.model_dump() for s in script_scenes]
    return _msgs(f"""Break the screenplay below down into production scenes.
{language_rule(language)}

Logline: {logline}
Known character names (use EXACTLY these spellings): {json.dumps(names, ensure_ascii=False)}
Target film duration: {duration_minutes * 60:.0f} seconds in total.
Constraints: {constraints or 'none'}
Hard limits:
{_limits_text(max_locations, max_actors)}
{('User instruction: ' + instructions) if instructions else ''}

Screenplay scenes:
{json.dumps(script, ensure_ascii=False, indent=1)}

Rules:
- Produce one entry per screenplay scene, same numbering (1, 2, 3, ...).
- Write the same place with the SAME location string every time.
- "estimated_duration_sec" must be a positive number; all scenes together should add up to about the target duration.
- "props" = physical objects that matter in the scene.
- time_of_day must be one of: {_enum_values(TimeOfDay)}

Return JSON with exactly this shape:
{{"scenes": [{{"number": 1, "heading": "...", "location": "...", "time_of_day": "NIGHT",
  "characters": ["..."], "props": ["..."], "action": "what happens, 1-3 sentences", "estimated_duration_sec": 40}}]}}""")


def create_shot_list_messages(*, scenes: Iterable[Scene], characters: Iterable[Character],
                              visual_style, equipment, constraints, language,
                              instructions="") -> list[dict]:
    scene_data = [
        {"number": s.number, "location": s.location, "time_of_day": s.time_of_day.value,
         "characters": s.characters, "props": s.props, "action": s.action,
         "estimated_duration_sec": s.estimated_duration_sec}
        for s in scenes
    ]
    details = {c.name: c.details for c in characters if c.details}
    gear = ", ".join(equipment) if equipment else "not specified (assume a single camera and a tripod)"
    return _msgs(f"""Create a shot list for the scenes below.
{language_rule(language)}

Visual style: {visual_style or 'not specified'}
Available equipment: {gear}
Constraints: {constraints or 'none'}
Fixed character details (keep consistent): {json.dumps(details, ensure_ascii=False)}
{('User instruction: ' + instructions) if instructions else ''}

Scenes:
{json.dumps(scene_data, ensure_ascii=False, indent=1)}

Rules:
- 2-5 shots per scene. Each shot's "scene_number" MUST be one of the scene numbers above.
- Number shots 1, 2, 3, ... in filming order across the whole film.
- "estimated_duration_sec" is positive; shots of a scene should add up to roughly the scene duration.
- Only use camera movements that the available equipment allows.
- shot_type: {_enum_values(ShotType)}
- camera_angle: {_enum_values(CameraAngle)}
- movement: {_enum_values(CameraMovement)}

Return JSON with exactly this shape:
{{"shots": [{{"number": 1, "scene_number": 1, "shot_type": "WIDE", "camera_angle": "EYE_LEVEL",
  "movement": "STATIC", "description": "what we see", "audio_notes": "sound, dialogue, music", "estimated_duration_sec": 8}}]}}""")


def semantic_review_messages(*, characters: Iterable[Character], scenes: Iterable[Scene],
                             shots, language) -> list[dict]:
    payload = {
        "characters": [c.model_dump() for c in characters],
        "scenes": [s.model_dump(mode="json") for s in scenes],
        "shots": [s.model_dump(mode="json") for s in shots],
    }
    return _msgs(f"""Review this short-film plan for CONTINUITY problems that simple rules cannot detect
(character costume/appearance changes, props appearing or vanishing, time-of-day or weather jumps,
actions that contradict each other). Report only real problems you can point to; if there are none, return an empty list.
You are giving suggestions, not guarantees.
{language_rule(language)}

Plan:
{json.dumps(payload, ensure_ascii=False)}

Return JSON with exactly this shape (at most 5 issues):
{{"issues": [{{"severity": "WARNING | INFO", "message": "what is inconsistent", "suggestion": "how to fix it", "scene_number": 1}}]}}""")


# --------------------------------------------------------------------------- #
# Orchestrator prompts
# --------------------------------------------------------------------------- #
def router_messages(*, user_message: str, project_summary: str, tools_description: str) -> list[dict]:
    return _msgs(f"""Decide which registered tool (if any) should handle the user's message.

Project state:
{project_summary}

Registered tools (you may ONLY choose from these names):
{tools_description}

Intents: DEVELOP_IDEA, BUILD_SCRIPT, BREAKDOWN_SCENES, CREATE_SHOT_LIST, PLAN_PRODUCTION,
CHECK_CONTINUITY, EXPORT_PACKAGE, GENERAL_QUESTION (no tool: a question or chat).

User message: {user_message}

Return JSON with exactly this shape:
{{"intent": "GENERAL_QUESTION", "tool": null, "arguments": {{}}, "missing_info": []}}
- "tool" is a registered tool name, or null for GENERAL_QUESTION.
- "arguments" may only contain "instructions" (the user's change request, copied from the message) when the tool accepts it.
- "missing_info" lists information you need from the user before the tool can run.""")


def general_answer_messages(*, project_summary: str, history: list[dict], user_message: str,
                            language: str) -> list[dict]:
    system = (
        "You are FRAME, a friendly film pre-production mentor for beginner filmmakers. "
        "Answer concisely and practically. If you do not know something, say so; "
        "never invent real-world prices, statistics or research results. "
        + ("Answer in English. " if language == "en" else "Answer in Mongolian (Cyrillic); technical film terms may stay in English. ")
        + f"\n\nCurrent project (for context only):\n{project_summary}"
    )
    messages = [{"role": "system", "content": system}]
    for turn in history[-6:]:
        if turn.get("role") not in ("user", "assistant") or not turn.get("content"):
            continue
        if turn["role"] == "assistant" and len(messages) == 1:
            continue  # a conversation must open with a user turn (e.g. skip the app's greeting note)
        messages.append({"role": turn["role"], "content": turn["content"][:1500]})
    messages.append({"role": "user", "content": user_message})
    return messages
