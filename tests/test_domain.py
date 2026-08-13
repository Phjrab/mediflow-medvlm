from datetime import date
import unittest

from mediflow_medvlm import InferenceRequest, TaskType
from mediflow_medvlm.domain import ImageReference


class DomainContractTests(unittest.TestCase):
    def test_request_keeps_reproducibility_metadata(self) -> None:
        request = InferenceRequest(
            study_id="study-pseudonym-01",
            task=TaskType.VQA,
            current_images=(ImageReference(uri="file:///image.png", acquired_at=date(2026, 1, 1)),),
            question="Is there a visible abnormality?",
            prompt_version="vqa-2026-01",
            model_parameters={"temperature": 0},
        )

        self.assertEqual(request.study_id, "study-pseudonym-01")
        self.assertEqual(request.task, TaskType.VQA)
        self.assertEqual(request.model_parameters["temperature"], 0)
