import joblib
from fastapi import HTTPException

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load("models/random_forest_regressor_model.joblib")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    return _model
