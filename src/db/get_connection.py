import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Establish a connection to the Supabase database."""
    try:
        db_url = os.getenv("SUPABASE_DB_URL")
        sslmode = os.getenv("DB_SSL_MODE", "require")
        connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))

        if db_url:
            print("Connecting to database using SUPABASE_DB_URL")
            conn = psycopg2.connect(dsn=db_url, sslmode=sslmode, connect_timeout=connect_timeout)
        else:
            print(f"Connecting to database host: {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}")
            conn = psycopg2.connect(
                host = os.getenv("DB_HOST"),
                port = os.getenv("DB_PORT"),
                user = os.getenv("DB_USER"),
                password = os.getenv("DB_PASSWORD"),
                dbname = os.getenv("DB_NAME"),
                sslmode = sslmode,
                connect_timeout = connect_timeout,
            )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise e

