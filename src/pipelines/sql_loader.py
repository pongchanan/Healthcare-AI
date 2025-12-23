import duckdb
import pandas as pd
import os
import glob
from src.config import Config

def ingest_csvs():
    """
    Loads raw CSVs into DuckDB.
    """
    if os.path.exists(Config.SQL_DB_PATH):
        os.remove(Config.SQL_DB_PATH) # Reset DB

    conn = duckdb.connect(Config.SQL_DB_PATH)
    
    csv_files = glob.glob("data/raw/*.csv")
    
    if not csv_files:
        print(" [Offline] No CSV files found. Creating dummy patients table.")
        conn.execute("CREATE TABLE patients (id VARCHAR, name VARCHAR, age INTEGER, diagnosis VARCHAR)")
        conn.execute("INSERT INTO patients VALUES ('123', 'John Doe', 30, 'Flu'), ('456', 'Jane Smith', 45, 'Hypertension')")
    else:
        for file in csv_files:
            table_name = os.path.basename(file).split('.')[0]
            print(f" [Offline] Ingesting {file} into table '{table_name}'...")
            try:
                df = pd.read_csv(file)
                conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
            except Exception as e:
                print(f"Error reading {file}: {e}")

    # Explicitly create dummy patients table if not present, to support the agent's default query
    try:
         conn.execute("SELECT * FROM patients LIMIT 1")
    except:
         print(" [Offline] Creating dummy patients table.")
         conn.execute("CREATE TABLE patients (id VARCHAR, name VARCHAR, age INTEGER, diagnosis VARCHAR)")
         conn.execute("INSERT INTO patients VALUES ('123', 'John Doe', 30, 'Flu'), ('456', 'Jane Smith', 45, 'Hypertension')")
    
    conn.close()
    print(" [Offline] CSV Data ingested into SQL Database.")

if __name__ == "__main__":
    ingest_csvs()
