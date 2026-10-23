from datetime import date
import unittest
import json
import ast
from pathlib import Path

from mediflow_medvlm import InferenceRequest, TaskType
from mediflow_medvlm.domain import ImageReference


class DomainContractTests(unittest.TestCase):
    def test_python310_enum_string_json_and_identity_contract(self):
        self.assertEqual(str(TaskType.VQA), "vqa")
        self.assertEqual(f"{TaskType.TEXT_QA}", "text_qa")
        self.assertEqual(json.dumps(TaskType.VQA), '"vqa"')
        self.assertIs(TaskType("vqa"), TaskType.VQA)
        import mediflow_medvlm.domain as domain
        ast.parse(Path(domain.__file__).read_text(), feature_version=(3, 10))

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
