import pytest

@pytest.fixture
def sample_state():
    return {
        "patient_name":    "Sara",
        "patient_input":   "I have fever, severe headache and chills",
        "symptoms_list":   ["fever", "headache", "chills"],
        "predicted_disease": "Influenza",
        "api_disease_info":  "Influenza is a viral respiratory illness.",
        "precautions":       "1. Rest  2. Fluids  3. See a doctor",
        "ml_prediction":     "Influenza",
    }

@pytest.fixture
def empty_state():
    return {
        "patient_name":  "TestPatient",
        "patient_input": "",
    }
