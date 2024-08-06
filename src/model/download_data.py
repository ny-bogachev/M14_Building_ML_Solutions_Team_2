# scripts/download_data.py

from sqlalchemy import create_engine
import pandas as pd

database_url = "postgresql://postgres.aeootalfoqhilupvmfac:xWbhAokb54DJdTb@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"

engine = create_engine(database_url)

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

df = pd.read_sql(query, engine)

df.to_csv('data/trips_2024_07_02.csv', index=False)
