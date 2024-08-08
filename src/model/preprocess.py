import pandas as pd
import holidays

def check_holiday_weekend(date):
    ny_holidays = holidays.US(state='NY')
    is_weekend = date.weekday() >= 5
    is_holiday = date in ny_holidays
    return pd.Series([is_weekend, is_holiday])

def preprocess(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    df_filtered = df[(df['trip_distance'] > 0) & 
                     (df['fare_amount'] >= 0) & 
                     (df['extra'] >= 0) & 
                     (df['mta_tax'] >= 0) & 
                     (df['tolls_amount'] >= 0) & 
                     (df['improvement_surcharge'] >= 0) & 
                     (df['total_amount'] >= 0) & 
                     (df['congestion_surcharge'] >= 0) & 
                     (df['Airport_fee'] >= 0) &
                     (df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime'])]

    df_filtered = df_filtered.reset_index(drop=True)
    
    df_filtered.loc[:, 'duration_sec'] = (df_filtered['tpep_dropoff_datetime'] - df_filtered['tpep_pickup_datetime']).dt.total_seconds()
    
    df_filtered.loc[:, 'congestion_surcharge_dummy'] = (df_filtered['congestion_surcharge'] > 0).astype(int)
    df_filtered.loc[:, 'airport_fee_dummy'] = (df_filtered['Airport_fee'] > 0).astype(int)
    
    df_filtered[['is_weekend', 'is_holiday']] = df_filtered['tpep_pickup_datetime'].apply(check_holiday_weekend)
    df_filtered.loc[:, 'is_weekend'] = df_filtered['is_weekend'].astype(int)
    df_filtered.loc[:, 'is_holiday'] = df_filtered['is_holiday'].astype(int)
    
    feature_columns = ['trip_distance', 'is_weekend', 'is_holiday', 
                       'congestion_surcharge_dummy', 'airport_fee_dummy']
    X = df_filtered[feature_columns]
    y = df_filtered['duration_sec']
    
    return X, y
