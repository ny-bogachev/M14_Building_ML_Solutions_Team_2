from typing import List
import pandas as pd
from fastapi import FastAPI, Body, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from .get_model import get_model

app = FastAPI()

class PredictRequest(BaseModel):
    tpep_pickup_datetime: str
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int

class PredictResponse(BaseModel):
    prediction: List[float]

def check_holiday_weekend(date):
    # Implement holiday and weekend check logic
    is_weekend = date.weekday() >= 5
    ny_holidays = holidays.US(state='NY')
    is_holiday = date in ny_holidays
    return pd.Series([is_weekend, is_holiday])

def preprocess_request(data: List[PredictRequest]):
    # Convert to DataFrame
    df = pd.DataFrame([{
        "tpep_pickup_datetime": item.tpep_pickup_datetime,
        "pickup_longitude": item.pickup_longitude,
        "pickup_latitude": item.pickup_latitude,
        "dropoff_longitude": item.dropoff_longitude,
        "dropoff_latitude": item.dropoff_latitude,
        "passenger_count": item.passenger_count
    } for item in data])

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
    df['trip_distance'] = ((df['dropoff_longitude'] - df['pickup_longitude'])**2 + (df['dropoff_latitude'] - df['pickup_latitude'])**2)**0.5
    df['duration_sec'] = (datetime.now() - df['tpep_pickup_datetime']).dt.total_seconds()  # Placeholder for actual calculation

    df['congestion_surcharge_dummy'] = 0
    df['airport_fee_dummy'] = 0
    df[['is_weekend', 'is_holiday']] = df['tpep_pickup_datetime'].apply(check_holiday_weekend)
    
    feature_columns = ['trip_distance', 'is_weekend', 'is_holiday', 'congestion_surcharge_dummy', 'airport_fee_dummy']
    input_df = df[feature_columns]
    
    return input_df

@app.post("/predict", response_model=PredictResponse)
def predict(
    data: List[PredictRequest] = Body(...),
    model = Depends(get_model)
) -> PredictResponse:
    try:
        input_df = preprocess_request(data)
        predictions = model.predict(input_df)
        return PredictResponse(prediction=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
