"""Safety-oriented task orchestration for medical VLM outputs."""

from __future__ import annotations

from collections.abc import Iterable

from .adapters import MedicalVLM
from .domain import InferenceRequest, InferenceResult, TaskType
from .validation import RequestValidationError, validate_request


_BASE_INSTRUCTION = """You are a research-only medical image analysis assistant.
Do not claim clinical certainty, make treatment recommendations, or infer facts
that are not visible in the supplied imaging context. Separate limitations from
findings. Return only the requested structured fields."""

_REQUIRED_FIELDS: dict[TaskType, tuple[str, ...]] = {
    TaskType.TEXT_QA: ("answer", "status", "limitations"),
    TaskType.VQA: ("answer", "evidence", "limitations"),
    TaskType.REPORT: ("findings", "impression", "limitations"),
    TaskType.LONGITUDINAL_COMPARISON: ("changes", "comparison_basis", "limitations"),
}


def run_workflow(adapter: MedicalVLM, request: InferenceRequest) -> InferenceResult:
    """Validate, execute, and validate the structured result of one task."""

    validate_request(request)
    result = adapter.generate(request, _instruction_for(request.task))
    _validate_result(request, result)
    return result


def _instruction_for(task: TaskType) -> str:
    if task is TaskType.TEXT_QA:
        return ("You are a research-only medical education assistant. Answer the educational "
                "question using the supplied context and medical knowledge. This is not personal "
                "diagnosis or prescription. Do not invent source evidence or rationale. "
                "Return answer, status, and limitations; include option_id for multiple choice. "
                "Use status answered or abstained.")
    task_contract = {
        TaskType.VQA: "Return answer, evidence, and limitations.",
        TaskType.REPORT: "Return findings, impression, and limitations.",
        TaskType.LONGITUDINAL_COMPARISON: (
            "Return changes, comparison_basis, and limitations. Do not compare studies "
            "unless the supplied imaging supports a valid comparison."
        ),
    }[task]
    return f"{_BASE_INSTRUCTION}\n{task_contract}"


def _validate_result(request: InferenceRequest, result: InferenceResult) -> None:
    if result.prompt_version != request.prompt_version:
        raise RequestValidationError("result prompt_version must match the request")
    if result.latency_ms < 0:
        raise RequestValidationError("result latency_ms cannot be negative")
    _require_nonempty_fields(result.content, _REQUIRED_FIELDS[request.task])
    if request.task is TaskType.TEXT_QA:
        if result.content["status"] not in {"answered", "abstained"}:
            raise RequestValidationError("TEXT_QA status must be answered or abstained")
        if request.options and result.content["status"] == "answered" and result.content.get("option_id") not in request.options:
            raise RequestValidationError("TEXT_QA option_id must identify a supplied option")


def _require_nonempty_fields(content: dict[str, object], required: Iterable[str]) -> None:
    missing = [field for field in required if not content.get(field)]
    if missing:
        raise RequestValidationError(f"model result is missing required fields: {', '.join(missing)}")
