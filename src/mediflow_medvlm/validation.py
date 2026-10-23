"""Input-policy checks executed before prompts or model providers are called."""

from __future__ import annotations

import re

from .domain import InferenceRequest, TaskType


class RequestValidationError(ValueError):
    """Raised when an input violates the public workflow contract."""


_DISALLOWED_IDENTIFIER_FIELDS = {
    "patient_name",
    "patient_id",
    "mrn",
    "date_of_birth",
    "dob",
    "dicom_uid",
    "accession_number",
}
_SAFE_URI_PREFIXES = ("file://", "s3://", "https://")
_MRN_LIKE_PATTERN = re.compile(r"\b(?:mrn|patient\s*id)\s*[:#]?\s*\w+", re.IGNORECASE)


def validate_request(request: InferenceRequest) -> None:
    """Validate task requirements and reject obvious PHI-bearing parameters.

    This is a boundary guard, not a substitute for institution-approved DICOM
    de-identification.  A production ingestion layer must remove PHI before an
    ``InferenceRequest`` can be constructed.
    """

    if not request.study_id or request.study_id.strip() != request.study_id:
        raise RequestValidationError("study_id must be a non-empty pseudonymous identifier")
    if not isinstance(request.task, TaskType):
        raise RequestValidationError("task must be a TaskType")
    if request.task is not TaskType.TEXT_QA and not request.current_images:
        raise RequestValidationError("at least one current image is required")
    if request.task in (TaskType.VQA, TaskType.TEXT_QA) and not (request.question or "").strip():
        raise RequestValidationError(f"a {request.task.value.upper()} request requires a question")
    if request.task is TaskType.TEXT_QA and (request.current_images or request.prior_images):
        raise RequestValidationError("TEXT_QA does not accept images; use VQA for image input")
    if not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in request.options.items()):
        raise RequestValidationError("options must map string identifiers to non-empty text")
    if request.task is TaskType.LONGITUDINAL_COMPARISON and not request.prior_images:
        raise RequestValidationError("longitudinal comparison requires at least one prior image")

    for image in (*request.current_images, *request.prior_images):
        if not image.uri.startswith(_SAFE_URI_PREFIXES):
            raise RequestValidationError("image uri must use file://, s3://, or https://")

    disallowed = _DISALLOWED_IDENTIFIER_FIELDS.intersection(request.model_parameters)
    if disallowed:
        fields = ", ".join(sorted(disallowed))
        raise RequestValidationError(f"model_parameters must not include direct identifiers: {fields}")

    if _MRN_LIKE_PATTERN.search(" ".join([request.question or "", request.context or "", *request.options.values()])):
        raise RequestValidationError("question appears to contain a direct patient identifier")
