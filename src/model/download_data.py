# scripts/download_data.py

import os
import pandas as pd
from sqlalchemy import create_engine
import yaml

# Load parameters from params.yaml
with open('params.yaml', 'r') as file: 
    params = yaml.safe_load(file)

data_dir = params['data_dir']
print("PATHHH:", data_dir)
csv_file_path = os.path.join(data_dir, 'trips_2024_07_02.csv')

if not os.path.exists(csv_file_path):
    print("CSV file not found. Downloading data...")

    # Define the database connection URL
    database_url = "postgresql://postgres.aeootalfoqhilupvmfac:xWbhAokb54DJdTb@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"

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

    # Save the data to a CSV file
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    df.to_csv(csv_file_path, index=False)
    
    print(f"Data downloaded and saved to {csv_file_path}")
else:
    print(f"CSV file already exists at {csv_file_path}. Skipping download.")
