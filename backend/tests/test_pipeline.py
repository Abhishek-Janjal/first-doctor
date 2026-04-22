from unittest.mock import patch, MagicMock
import numpy as np

from backend.agents.PredDoc import medical_assessment_graph

def test_full_pipeline_runs():
    """
    End-to-end test — mocks everything external.
    Proves the agents wiring is correct (all nodes connect, state flows through).
    """
    mock_api_response = MagicMock()
    mock_api_response.status_code = 200
    mock_api_response.json.return_value = {"extract": "Influenza is a viral illness."}

    with patch("agents.nodes._llm_call") as mock_llm, \
         patch("agents.nodes.KNN_MODEL") as mock_knn, \
         patch("agents.nodes.LABEL_ENCODER") as mock_le, \
         patch("agents.nodes.requests.get", return_value=mock_api_response):

        # Every LLM call returns something sensible
        mock_llm.side_effect = [
            '["fever", "headache", "chills"]',  # extract_symptoms
            "Influenza",                          # predict_disease
            "1. Rest\n2. Drink fluids",           # suggest_precautions
            "Dear Sara, you have Influenza.",     # orchestrate_summary
        ]
        mock_knn.predict.return_value = np.array([0])
        mock_le.inverse_transform.return_value = np.array(["Influenza"])

        result = medical_assessment_graph.invoke({
            "patient_name":  "Sara",
            "patient_input": "I have fever, headache, and chills",
        })

    # Every expected key is in the final state
    assert result["symptoms_list"]     == ["fever", "headache", "chills"]
    assert result["predicted_disease"] == "Influenza"
    assert result["precautions"]       is not None
    assert result["final_summary"]     is not None

def test_pipeline_state_flows_correctly():
    """Each node's output becomes the next node's input."""
    mock_api = MagicMock()
    mock_api.status_code = 200
    mock_api.json.return_value = {"extract": "Gastroenteritis info."}

    with patch("agents.nodes._llm_call") as mock_llm, \
         patch("agents.nodes.KNN_MODEL") as mock_knn, \
         patch("agents.nodes.LABEL_ENCODER") as mock_le, \
         patch("agents.nodes.requests.get", return_value=mock_api):

        mock_llm.side_effect = [
            '["nausea", "vomiting", "diarrhea"]',
            "Gastroenteritis",
            "1. Stay hydrated",
            "Ahmed, you have Gastroenteritis.",
        ]
        mock_knn.predict.return_value = np.array([1])
        mock_le.inverse_transform.return_value = np.array(["Gastroenteritis"])

        result = medical_assessment_graph.invoke({
            "patient_name":  "Ahmed",
            "patient_input": "Nausea, vomiting and diarrhea",
        })

    assert "nausea" in result["symptoms_list"]
    assert result["predicted_disease"] == "Gastroenteritis"
