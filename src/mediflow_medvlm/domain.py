"""Stable data contracts shared by all model adapters and workflows.

The contracts deliberately exclude direct patient identifiers.  Callers should
pass a pseudonymous study identifier and must keep the mapping inside their
approved clinical or research environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


class TaskType(StrEnum):
    """Supported high-level medical VLM tasks."""

    VQA = "vqa"
    REPORT = "report"
    LONGITUDINAL_COMPARISON = "longitudinal_comparison"


@dataclass(frozen=True, slots=True)
class ImageReference:
    """A non-PHI reference to one validated image or series asset."""

    uri: str
    acquired_at: date | None = None
    modality: str | None = None
    body_part: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Input to a model adapter after ingestion and policy validation."""

    study_id: str
    task: TaskType
    current_images: tuple[ImageReference, ...]
    question: str | None = None
    prior_images: tuple[ImageReference, ...] = ()
    prompt_version: str = "unversioned"
    model_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """A model result plus the data required to reproduce and audit it."""

    content: dict[str, Any]
    model_id: str
    model_revision: str | None
    prompt_version: str
    latency_ms: int
    warnings: tuple[str, ...] = ()
