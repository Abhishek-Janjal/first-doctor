from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# ── Test _parse_symptoms (pure function — no mocking needed) ──────────────────

from backend.agents.PredDoc.nodes import _parse_symptoms

def test_parse_symptoms_valid_list():
    raw = '["fever", "headache", "nausea"]'
    result = _parse_symptoms(raw)
    assert result == ["fever", "headache", "nausea"]

def test_parse_symptoms_no_brackets():
    # LLM sometimes returns without brackets
    raw = "fever, headache, nausea"
    result = _parse_symptoms(raw)
    assert "fever" in result
    assert len(result) == 3

def test_parse_symptoms_empty_string():
    result = _parse_symptoms("")
    assert result == []

def test_parse_symptoms_with_llm_noise():
    # LLM sometimes adds text before the list
    raw = 'Here are the symptoms: ["cough", "fever"]'
    result = _parse_symptoms(raw)
    assert result == ["cough", "fever"]


# ── Test _fuzzy_map (uses real data — no mocking needed) ──────────────────────

from backend.agents.PredDoc.nodes import _fuzzy_map

def test_fuzzy_map_exact_match():
    # "fever" should map to itself if it's in the dataset
    result = _fuzzy_map("fever")
    assert result is not None  # found something
    assert isinstance(result, str)

def test_fuzzy_map_close_match():
    # Typo — should still find the right symptom
    result = _fuzzy_map("fevver")   # typo
    assert result is not None

def test_fuzzy_map_garbage_returns_none():
    result = _fuzzy_map("xyzabc123nonsense")
    assert result is None


# ── Test extract_symptoms (mocks the LLM) ─────────────────────────────────────

from backend.agents.PredDoc.nodes import extract_symptoms

def test_extract_symptoms_happy_path():
    state = {"patient_input": "I have fever and headache"}

    with patch("agents.nodes._llm_call") as mock_llm:
        mock_llm.return_value = '["fever", "headache"]'
        result = extract_symptoms(state)

    assert "symptoms_list" in result
    assert "fever" in result["symptoms_list"]
    assert "headache" in result["symptoms_list"]

def test_extract_symptoms_empty_input():
    state = {"patient_input": ""}

    with patch("agents.nodes._llm_call") as mock_llm:
        mock_llm.return_value = "[]"
        result = extract_symptoms(state)

    assert result["symptoms_list"] == []

def test_extract_symptoms_missing_key():
    # No "patient_input" key at all — should not KeyError
    state = {}
    with patch("agents.nodes._llm_call") as mock_llm:
        mock_llm.return_value = "[]"
        # Should return gracefully, not crash
        try:
            result = extract_symptoms(state)
            assert "symptoms_list" in result or "error" in result
        except KeyError:
            pytest.fail("extract_symptoms raised KeyError — fix the .get() issue!")


# ── Test predict_disease (mocks LLM + ML model) ───────────────────────────────

from backend.agents.PredDoc.nodes import predict_disease

def test_predict_disease_with_valid_symptoms(sample_state):
    with patch("agents.nodes._llm_call") as mock_llm, \
         patch("agents.nodes.KNN_MODEL") as mock_knn, \
         patch("agents.nodes.LABEL_ENCODER") as mock_le:

        mock_knn.predict.return_value = np.array([0])
        mock_le.inverse_transform.return_value = np.array(["Influenza"])
        mock_llm.return_value = "Influenza"

        result = predict_disease(sample_state)

    assert "predicted_disease" in result
    assert "ml_prediction" in result
    assert result["predicted_disease"] == "Influenza"

def test_predict_disease_no_symptoms():
    state = {"symptoms_list": []}
    with patch("agents.nodes._llm_call") as mock_llm:
        mock_llm.return_value = "Unknown condition"
        result = predict_disease(state)

    assert result["ml_prediction"] == "Unknown — symptoms not recognised in dataset"


# ── Test fetch_disease_info (mocks HTTP request) ──────────────────────────────

from backend.agents.PredDoc.nodes import fetch_disease_info

def test_fetch_disease_info_success(sample_state):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "extract": "Influenza is a contagious respiratory illness caused by viruses."
    }

    with patch("agents.nodes.requests.get", return_value=mock_response):
        result = fetch_disease_info(sample_state)

    assert "api_disease_info" in result
    assert "Influenza" in result["api_disease_info"]

def test_fetch_disease_info_api_failure(sample_state):
    with patch("agents.nodes.requests.get", side_effect=Exception("Network error")):
        result = fetch_disease_info(sample_state)

    assert "api_disease_info" in result
    assert "unavailable" in result["api_disease_info"].lower()  # graceful fallback

def test_fetch_disease_info_404(sample_state):
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("agents.nodes.requests.get", return_value=mock_response):
        result = fetch_disease_info(sample_state)

    assert "404" in result["api_disease_info"] or "local knowledge" in result["api_disease_info"]


# ── Test suggest_precautions (mocks LLM) ─────────────────────────────────────

from backend.agents.PredDoc.nodes import suggest_precautions

def test_suggest_precautions_known_disease(sample_state):
    with patch("agents.nodes._llm_call") as mock_llm:
        mock_llm.return_value = "1. Rest\n2. Fluids\n3. See doctor\n4. Avoid contact\n5. Monitor temp"
        result = suggest_precautions(sample_state)

    assert "precautions" in result
    assert len(result["precautions"]) > 0

def test_suggest_precautions_unknown_disease():
    state = {
        "predicted_disease": "SomeRareUnknownDisease99",
        "api_disease_info": ""
    }
    with patch("agents.nodes._llm_call") as mock_llm:
        mock_llm.return_value = "1. Rest\n2. See a doctor"
        result = suggest_precautions(state)

    assert "precautions" in result


# ── Test orchestrate_summary ──────────────────────────────────────────────────

from backend.agents.PredDoc.nodes import orchestrate_summary

def test_orchestrate_summary_produces_report(sample_state):
    with patch("agents.nodes._llm_call") as mock_llm:
        mock_llm.return_value = "Dear Sara, based on your symptoms you have Influenza. Please rest."
        result = orchestrate_summary(sample_state)

    assert "final_summary" in result
    assert len(result["final_summary"]) > 20

def test_orchestrate_summary_uses_patient_name(sample_state):
    with patch("agents.nodes._llm_call") as mock_llm:
        mock_llm.return_value = "Dear Sara, you have Influenza."
        result = orchestrate_summary(sample_state)

    # The LLM received the patient name (we can check the call args)
    call_args = mock_llm.call_args
    assert "Sara" in call_args[0][1]   # "Sara" appears in the human prompt
