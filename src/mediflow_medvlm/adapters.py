"""Provider-neutral model adapter contract."""

from __future__ import annotations

from typing import Protocol

from .domain import InferenceRequest, InferenceResult


class MedicalVLM(Protocol):
    """Any local or remote model implementation usable by a workflow."""

    def generate(self, request: InferenceRequest, system_instruction: str) -> InferenceResult:
        """Generate a structured response for an already-validated request."""
