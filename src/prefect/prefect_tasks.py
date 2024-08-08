import os
import pandas as pd
from sqlalchemy import create_engine
import yaml
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_error
from prefect import task, get_run_logger
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

@task
def load_params(params_path: str):
    """Load parameters from a YAML file."""
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

@task
def check_file_exists(file_path: str):
    """Check if the CSV file exists."""
    return os.path.exists(file_path)

@task
def download_data(params: dict):
    """Download data from the database if the CSV file does not exist."""
    logger = get_run_logger()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    csv_file_path = os.path.join(data_dir, 'trips_2024_07_02.csv')

    if not os.path.exists(csv_file_path):
        logger.info("CSV file not found. Downloading data...")
        
        # Define the database connection URL
        database_url = params['database_url']
        
        # Create a database engine
        engine = create_engine(database_url)
        
        # Define the SQL query
        query = """
        SELECT
          *
        FROM
          public.trips_2024_07_02
        ORDER BY
          tpep_dropoff_datetime DESC
        LIMIT
          100000;
        """
        
        # Fetch the data from the database
        df = pd.read_sql(query, engine)
        
        # Create data directory if it does not exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Save the data to a CSV file
        df.to_csv(csv_file_path, index=False)
        
        logger.info(f"Data downloaded and saved to {csv_file_path}")
    else:
        logger.info(f"CSV file already exists at {csv_file_path}. Skipping download.")

@task
def preprocess_data(params: dict):
    """Preprocess the data."""
    logger = get_run_logger()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    data_file_path = os.path.join(data_dir, 'trips_2024_07_02.csv')
    preprocessed_file_path = os.path.join(data_dir, 'preprocessed_data.csv')
    
    # Load the data
    df = pd.read_csv(data_file_path)
    
    # Apply preprocessing
    X, y = preprocess(df)
    
    # Combine features and target into one DataFrame for saving
    preprocessed_df = X.copy()
    preprocessed_df['duration_sec'] = y
    
    preprocessed_df.to_csv(preprocessed_file_path, index=False)
    
    logger.info(f"Data preprocessed and saved to {preprocessed_file_path}")
    return preprocessed_file_path

@task
def train_model(preprocessed_file_path: str):
    """Train the model."""
    logger = get_run_logger()
    
    # Load the preprocessed data
    df = pd.read_csv(preprocessed_file_path)
    X = df.drop(columns=['duration_sec'])
    y = df['duration_sec']
    
    # Train the model
    train_size = int(len(X) * 0.9)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib_file = os.path.join(model_dir, "random_forest_regressor_model.joblib")
    joblib.dump(model, joblib_file)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    logger.info(f"Model saved to {joblib_file}")
    logger.info(f"Model MSE: {mse}")
