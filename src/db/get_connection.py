import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Establish a connection to the Supabase database."""
    try:
        db_url = os.getenv("SUPABASE_DB_URL")
        print(f"Connecting to database with URL: {db_url}")  # Debugging
        conn = psycopg2.connect(
            host = os.getenv("DB_HOST"),  # Force IPv4
            port = 5432,
            user = os.getenv("DB_USER"),
            password = os.getenv("DB_PASSWORD"),
            dbname = os.getenv("DB_NAME"),
            # pool_mode= 'session'
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise e

