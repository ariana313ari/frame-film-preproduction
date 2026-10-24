"""Pydantic data models for FRAME.

The `Project` model is the single authoritative project state. The chat transcript
only holds text; every structured artifact (script, scenes, shots, plan, issues)
lives here, together with a per-stage freshness status so that dependent artifacts
can be marked OUTDATED when an upstream artifact is regenerated.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Enums (tolerant of the casing / spelling variations small LLMs produce)
# --------------------------------------------------------------------------- #
def _coerce_enum(cls: type[Enum], value: Any, aliases: dict[str, str]) -> Enum | None:
    """Map 'close-up', 'CloseUp', 'Close Up' ... onto an enum member, else None."""
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s\-/.]+", "_", value.strip().upper()).strip("_")
    key = aliases.get(key, key)
    return cls.__members__.get(key)


class ShotType(str, Enum):
    ESTABLISHING = "ESTABLISHING"
    WIDE = "WIDE"
    MEDIUM = "MEDIUM"
    TWO_SHOT = "TWO_SHOT"
    OVER_THE_SHOULDER = "OVER_THE_SHOULDER"
    CLOSE_UP = "CLOSE_UP"
    EXTREME_CLOSE_UP = "EXTREME_CLOSE_UP"
    INSERT = "INSERT"
    POV = "POV"

    @classmethod
    def _missing_(cls, value):
        return _coerce_enum(cls, value, _SHOT_TYPE_ALIASES)


_SHOT_TYPE_ALIASES = {
    "CLOSEUP": "CLOSE_UP", "CU": "CLOSE_UP", "CLOSE": "CLOSE_UP", "MEDIUM_CLOSE_UP": "CLOSE_UP",
    "ECU": "EXTREME_CLOSE_UP", "EXTREME_CLOSEUP": "EXTREME_CLOSE_UP",
    "WS": "WIDE", "WIDE_SHOT": "WIDE", "LONG_SHOT": "WIDE", "FULL_SHOT": "WIDE", "LONG": "WIDE",
    "MS": "MEDIUM", "MEDIUM_SHOT": "MEDIUM", "MID_SHOT": "MEDIUM",
    "OTS": "OVER_THE_SHOULDER", "OVER_SHOULDER": "OVER_THE_SHOULDER",
    "ESTABLISHING_SHOT": "ESTABLISHING", "INSERT_SHOT": "INSERT", "2_SHOT": "TWO_SHOT",
    "POINT_OF_VIEW": "POV", "POV_SHOT": "POV",
}


class CameraAngle(str, Enum):
    EYE_LEVEL = "EYE_LEVEL"
    LOW = "LOW"
    HIGH = "HIGH"
    DUTCH = "DUTCH"
    OVERHEAD = "OVERHEAD"

    @classmethod
    def _missing_(cls, value):
        return _coerce_enum(cls, value, _ANGLE_ALIASES)


_ANGLE_ALIASES = {
    "EYE": "EYE_LEVEL", "EYELEVEL": "EYE_LEVEL", "NEUTRAL": "EYE_LEVEL", "LEVEL": "EYE_LEVEL",
    "LOW_ANGLE": "LOW", "HIGH_ANGLE": "HIGH", "DUTCH_ANGLE": "DUTCH", "TILTED": "DUTCH",
    "BIRDS_EYE": "OVERHEAD", "TOP_DOWN": "OVERHEAD", "TOP": "OVERHEAD",
}


class CameraMovement(str, Enum):
    STATIC = "STATIC"
    PAN = "PAN"
    TILT = "TILT"
    DOLLY = "DOLLY"
    TRACKING = "TRACKING"
    HANDHELD = "HANDHELD"
    GIMBAL = "GIMBAL"
    ZOOM = "ZOOM"
    CRANE = "CRANE"

    @classmethod
    def _missing_(cls, value):
        return _coerce_enum(cls, value, _MOVEMENT_ALIASES)


_MOVEMENT_ALIASES = {
    "NONE": "STATIC", "FIXED": "STATIC", "LOCKED_OFF": "STATIC", "LOCKED": "STATIC", "STILL": "STATIC",
    "TRACK": "TRACKING", "TRACKING_SHOT": "TRACKING", "DOLLY_IN": "DOLLY", "DOLLY_OUT": "DOLLY",
    "PUSH_IN": "DOLLY", "PULL_OUT": "DOLLY", "HAND_HELD": "HANDHELD", "STEADICAM": "GIMBAL",
    "ZOOM_IN": "ZOOM", "ZOOM_OUT": "ZOOM", "PAN_LEFT": "PAN", "PAN_RIGHT": "PAN",
    "TILT_UP": "TILT", "TILT_DOWN": "TILT", "JIB": "CRANE",
}


class TimeOfDay(str, Enum):
    DAY = "DAY"
    NIGHT = "NIGHT"
    DAWN = "DAWN"
    DUSK = "DUSK"
    UNSPECIFIED = "UNSPECIFIED"

    @classmethod
    def _missing_(cls, value):
        return _coerce_enum(cls, value, _TIME_ALIASES)


_TIME_ALIASES = {
    "": "UNSPECIFIED", "UNKNOWN": "UNSPECIFIED", "ANY": "UNSPECIFIED",
    "MORNING": "DAY", "AFTERNOON": "DAY", "DAYTIME": "DAY", "NOON": "DAY",
    "EVENING": "DUSK", "SUNSET": "DUSK", "TWILIGHT": "DUSK",
    "SUNRISE": "DAWN", "EARLY_MORNING": "DAWN",
    "NIGHTTIME": "NIGHT", "MIDNIGHT": "NIGHT", "LATE_NIGHT": "NIGHT",
    "ШӨНӨ": "NIGHT", "ӨДӨР": "DAY", "ӨГЛӨӨ": "DAY", "ОРОЙ": "DUSK", "ШИНГЭН": "DAWN",
}


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

    @classmethod
    def _missing_(cls, value):
        return _coerce_enum(cls, value, {"WARN": "WARNING", "CRITICAL": "ERROR", "HIGH": "ERROR",
                                         "MEDIUM": "WARNING", "LOW": "INFO", "NOTE": "INFO"})


class BudgetLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @classmethod
    def _missing_(cls, value):
        return _coerce_enum(cls, value, {"БАГА": "LOW", "ДУНД": "MEDIUM", "ИХ": "HIGH", "ӨНДӨР": "HIGH"})


class Stage(str, Enum):
    """Workflow stages, in dependency order."""
    IDEA = "idea"
    SCRIPT = "script"
    SCENES = "scenes"
    SHOTS = "shots"
    PLAN = "plan"
    VALIDATION = "validation"


class ArtifactStatus(str, Enum):
    EMPTY = "empty"
    FRESH = "fresh"
    OUTDATED = "outdated"


class Intent(str, Enum):
    DEVELOP_IDEA = "DEVELOP_IDEA"
    BUILD_SCRIPT = "BUILD_SCRIPT"
    BREAKDOWN_SCENES = "BREAKDOWN_SCENES"
    CREATE_SHOT_LIST = "CREATE_SHOT_LIST"
    PLAN_PRODUCTION = "PLAN_PRODUCTION"
    CHECK_CONTINUITY = "CHECK_CONTINUITY"
    EXPORT_PACKAGE = "EXPORT_PACKAGE"
    GENERAL_QUESTION = "GENERAL_QUESTION"

    @classmethod
    def _missing_(cls, value):
        return _coerce_enum(cls, value, {})


# Artifacts that become stale when the key stage is regenerated.
DOWNSTREAM: dict[Stage, tuple[Stage, ...]] = {
    Stage.IDEA: (),
    Stage.SCRIPT: (Stage.SCENES, Stage.SHOTS, Stage.PLAN, Stage.VALIDATION),
    Stage.SCENES: (Stage.SHOTS, Stage.PLAN, Stage.VALIDATION),
    Stage.SHOTS: (Stage.PLAN, Stage.VALIDATION),
    Stage.PLAN: (),
    Stage.VALIDATION: (),
}


# --------------------------------------------------------------------------- #
# Base model + list coercion
# --------------------------------------------------------------------------- #
def _listify(value: Any) -> Any:
    """LLMs sometimes return 'a, b' where a list is expected."""
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
    return value


class FrameModel(BaseModel):
    """Ignores unknown keys and treats explicit nulls as 'use the default'."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


# --------------------------------------------------------------------------- #
# Story / production models
# --------------------------------------------------------------------------- #
class Character(FrameModel):
    name: str = Field(min_length=1)
    role: str = ""
    description: str = ""
    details: list[str] = Field(default_factory=list)  # fixed visual details, e.g. "red raincoat"

    _coerce_details = field_validator("details", mode="before")(_listify)


class ScriptScene(FrameModel):
    """A scene as written in the screenplay (before structured breakdown)."""

    number: int = Field(ge=1)
    heading: str = ""
    text: str = Field(min_length=1)


class Scene(FrameModel):
    number: int = Field(ge=1)
    heading: str = ""
    location: str = Field(min_length=1)
    time_of_day: TimeOfDay = TimeOfDay.UNSPECIFIED
    characters: list[str] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    action: str = ""
    # Not constrained here on purpose: check_continuity reports non-positive values.
    estimated_duration_sec: float = 0

    _coerce_lists = field_validator("characters", "props", mode="before")(_listify)


class Shot(FrameModel):
    number: int = Field(ge=1)
    scene_number: int = Field(ge=1)
    shot_type: ShotType = ShotType.MEDIUM
    camera_angle: CameraAngle = CameraAngle.EYE_LEVEL
    movement: CameraMovement = CameraMovement.STATIC
    description: str = ""
    audio_notes: str = ""
    estimated_duration_sec: float = 0


class PropItem(FrameModel):
    name: str
    scene_numbers: list[int] = Field(default_factory=list)


class EquipmentItem(FrameModel):
    name: str
    reason: str = ""
    # True/False when the project lists its equipment, None when that is unknown.
    available: bool | None = None


class LocationItem(FrameModel):
    name: str
    scene_numbers: list[int] = Field(default_factory=list)
    time_of_day: str = ""


class ScheduleBlock(FrameModel):
    day: int = Field(ge=1)
    order: int = Field(ge=1)
    label: str
    scene_numbers: list[int] = Field(default_factory=list)
    estimated_minutes: int = Field(ge=0)
    notes: str = ""


class ProductionPlan(FrameModel):
    props: list[PropItem] = Field(default_factory=list)
    equipment: list[EquipmentItem] = Field(default_factory=list)
    locations: list[LocationItem] = Field(default_factory=list)
    schedule: list[ScheduleBlock] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @property
    def total_minutes(self) -> int:
        return sum(b.estimated_minutes for b in self.schedule)

    @property
    def shoot_days(self) -> int:
        return max((b.day for b in self.schedule), default=0)


class ValidationIssue(FrameModel):
    severity: Severity
    code: str = "GENERAL"
    message: str
    suggestion: str = ""
    # 'deterministic' = computed by Python rules, 'model' = LLM suggestion (not guaranteed).
    source: Literal["deterministic", "model"] = "deterministic"
    scene_number: int | None = None
    shot_number: int | None = None


class IdeaNotes(FrameModel):
    summary: str = ""
    logline_options: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    suggested_decisions: list[str] = Field(default_factory=list)
    story_questions: list[str] = Field(default_factory=list)

    _coerce_lists = field_validator(
        "logline_options", "themes", "suggested_decisions", "story_questions", mode="before"
    )(_listify)


# --------------------------------------------------------------------------- #
# Structured outputs requested from the model (validated before use)
# --------------------------------------------------------------------------- #
class ScriptOutput(FrameModel):
    logline: str = Field(min_length=1)
    synopsis: str = Field(min_length=1)
    characters: list[Character] = Field(min_length=1)
    scenes: list[ScriptScene] = Field(min_length=1)


class SceneBreakdownOutput(FrameModel):
    scenes: list[Scene] = Field(min_length=1)


class ShotListOutput(FrameModel):
    shots: list[Shot] = Field(min_length=1)


class ModelIssue(FrameModel):
    severity: Severity = Severity.WARNING
    message: str = Field(min_length=1)
    suggestion: str = ""
    scene_number: int | None = None


class SemanticReviewOutput(FrameModel):
    issues: list[ModelIssue] = Field(default_factory=list)


class RouterProposal(FrameModel):
    """What the model proposes for a user message. The orchestrator verifies it."""

    intent: Intent
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_info: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Tool result envelope
# --------------------------------------------------------------------------- #
class ToolResult(FrameModel):
    ok: bool
    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)  # e.g. {"origin": "model:qwen2.5:3b"}

    @classmethod
    def success(cls, tool: str, data: dict[str, Any] | None = None, *, warnings=None,
                issues=None, meta=None) -> "ToolResult":
        return cls(ok=True, tool=tool, data=data or {}, warnings=warnings or [],
                   issues=issues or [], meta=meta or {})

    @classmethod
    def failure(cls, tool: str, error: str, *, warnings=None) -> "ToolResult":
        return cls(ok=False, tool=tool, error=error, warnings=warnings or [])


# --------------------------------------------------------------------------- #
# Project: the single source of truth
# --------------------------------------------------------------------------- #
def _initial_status() -> dict[Stage, ArtifactStatus]:
    return {stage: ArtifactStatus.EMPTY for stage in Stage}


class Project(FrameModel):
    # Brief
    name: str = "Нэргүй төсөл"
    genre: str = ""
    duration_minutes: float = Field(3.0, gt=0, le=60)
    max_locations: int | None = Field(None, ge=1)
    max_actors: int | None = Field(None, ge=1)
    budget: BudgetLevel = BudgetLevel.LOW
    visual_style: str = ""
    constraints: str = ""
    equipment: list[str] = Field(default_factory=list)
    language: Literal["mn", "en"] = "mn"

    # Creative decisions
    idea: str = ""
    approved_decisions: list[str] = Field(default_factory=list)
    idea_notes: IdeaNotes | None = None

    # Artifacts
    logline: str = ""
    synopsis: str = ""
    characters: list[Character] = Field(default_factory=list)
    script_scenes: list[ScriptScene] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    plan: ProductionPlan | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)

    # Bookkeeping
    status: dict[Stage, ArtifactStatus] = Field(default_factory=_initial_status)
    provenance: dict[Stage, str] = Field(default_factory=dict)  # 'model:<name>' | 'sample' | 'deterministic'

    # -- status helpers ---------------------------------------------------- #
    def get_status(self, stage: Stage) -> ArtifactStatus:
        return self.status.get(stage, ArtifactStatus.EMPTY)

    def is_fresh(self, stage: Stage) -> bool:
        return self.get_status(stage) == ArtifactStatus.FRESH

    def _mark_fresh(self, stage: Stage, origin: str) -> None:
        self.status[stage] = ArtifactStatus.FRESH
        self.provenance[stage] = origin
        for dependent in DOWNSTREAM[stage]:
            if self.get_status(dependent) != ArtifactStatus.EMPTY:
                self.status[dependent] = ArtifactStatus.OUTDATED

    @property
    def duration_seconds(self) -> float:
        return self.duration_minutes * 60

    @property
    def total_scene_seconds(self) -> float:
        return sum(s.estimated_duration_sec for s in self.scenes)

    @property
    def is_blank(self) -> bool:
        return not (self.idea.strip() or self.script_scenes or self.scenes or self.shots)

    # -- state transitions (call only with validated results) --------------- #
    def apply_idea(self, notes: IdeaNotes, origin: str = "model") -> None:
        self.idea_notes = notes
        self._mark_fresh(Stage.IDEA, origin)

    def apply_script(self, out: ScriptOutput, origin: str = "model") -> None:
        self.logline = out.logline
        self.synopsis = out.synopsis
        self.characters = list(out.characters)
        self.script_scenes = list(out.scenes)
        self._mark_fresh(Stage.SCRIPT, origin)

    def apply_scenes(self, scenes: list[Scene], origin: str = "model") -> None:
        self.scenes = list(scenes)
        self._mark_fresh(Stage.SCENES, origin)

    def apply_shots(self, shots: list[Shot], origin: str = "model") -> None:
        self.shots = list(shots)
        self._mark_fresh(Stage.SHOTS, origin)

    def apply_plan(self, plan: ProductionPlan, origin: str = "deterministic") -> None:
        self.plan = plan
        self._mark_fresh(Stage.PLAN, origin)

    def apply_validation(self, issues: list[ValidationIssue], origin: str = "deterministic") -> None:
        self.issues = list(issues)
        self._mark_fresh(Stage.VALIDATION, origin)

    # -- prompt context ------------------------------------------------------ #
    def brief_text(self) -> str:
        parts = [
            f"Title: {self.name}",
            f"Genre: {self.genre or 'not set'}",
            f"Target duration: {self.duration_minutes:g} minutes",
            f"Max locations: {self.max_locations or 'not limited'}",
            f"Max actors: {self.max_actors or 'not limited'}",
            f"Budget level: {self.budget.value.lower()}",
            f"Visual style: {self.visual_style or 'not set'}",
            f"Other constraints: {self.constraints or 'none'}",
        ]
        return "\n".join(parts)

    def context_summary(self) -> str:
        lines = [self.brief_text(), f"Premise: {self.idea or 'not set'}"]
        if self.logline:
            lines.append(f"Logline: {self.logline}")
        if self.characters:
            lines.append("Characters: " + ", ".join(c.name for c in self.characters))
        lines.append(
            "Progress: " + ", ".join(f"{s.value}={self.get_status(s).value}" for s in Stage)
        )
        return "\n".join(lines)
