import pytest
from mediflow_medvlm.data.supplemental import medmcqa_records, join_aihub_71762


def test_index_base_explicit_and_unlabeled_test_never_scored():
    raw = {"id": "q1", "question": "Question?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 1, "choice_type": "single", "exp": None}
    kwargs = dict(revision="resolved", license_ref="verified-license", fetched_at="2026-10-23")
    rows, _ = medmcqa_records([raw], original_split="train", cop_base=1, **kwargs)
    assert rows[0]["supervision"]["option_id"] == "A"
    assert rows[0]["supervision"]["rationale"] is None
    rows, _ = medmcqa_records([raw], original_split="train", cop_base=0, **kwargs)
    assert rows[0]["supervision"]["option_id"] == "B"
    rows, excluded = medmcqa_records([raw], original_split="test", cop_base=0, **kwargs)
    assert not rows and excluded["official_test_labels_unavailable"] == 1


def test_unsafe_ambiguous_and_unmatched_aihub_pairing_rejected():
    with pytest.raises(ValueError, match="UNRESOLVED_PAIRING"):
        join_aihub_71762([{"disease": "x"}], [{"disease": "x"}], question_key="disease", answer_key="disease", guide_evidence="guide")
    with pytest.raises(ValueError, match="ambiguous"):
        join_aihub_71762([{"qid": 1}], [{"qid": 1}, {"qid": 1}], question_key="qid", answer_key="qid", guide_evidence="guide")
    with pytest.raises(ValueError, match="unmatched"):
        join_aihub_71762([{"qid": 1}], [{"qid": 2}], question_key="qid", answer_key="qid", guide_evidence="guide")
