"""Schema tests — verify Pydantic input/output models behave correctly."""
from api.schemas.churn_schemas import ChurnInput, ChurnOutput

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 45.50,
    "TotalCharges": "546.00",
}


def test_churn_input_accepts_valid_payload():
    obj = ChurnInput(**VALID_PAYLOAD)
    assert obj.tenure == 12
    assert obj.MonthlyCharges == 45.50


def test_churn_input_all_fields_optional():
    obj = ChurnInput()
    assert obj.gender is None
    assert obj.tenure is None


def test_churn_input_senior_citizen_accepts_int():
    obj = ChurnInput(SeniorCitizen=1)
    assert obj.SeniorCitizen == 1


def test_churn_output_valid():
    out = ChurnOutput(prediction="Yes", probability=0.85, model_version="0.4.0")
    assert out.prediction == "Yes"
    assert 0.0 <= out.probability <= 1.0


def test_churn_output_default_version():
    out = ChurnOutput(prediction="No", probability=0.1)
    assert out.model_version == "0.4.0"


def test_churn_output_prediction_values():
    yes = ChurnOutput(prediction="Yes", probability=0.9)
    no = ChurnOutput(prediction="No", probability=0.1)
    assert yes.prediction == "Yes"
    assert no.prediction == "No"


def test_churn_input_monthly_charges_float():
    obj = ChurnInput(MonthlyCharges=99.99)
    assert isinstance(obj.MonthlyCharges, float)


def test_churn_input_example_from_schema():
    example = ChurnInput.model_config["json_schema_extra"]["example"]
    obj = ChurnInput(**example)
    assert obj.gender == "Female"
    assert obj.Contract == "Month-to-month"
