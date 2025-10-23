import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Establish a connection to the Supabase database."""
    try:
        sslmode = os.getenv("DB_SSL_MODE", "require")
        connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))
        
        # Prioritize individual connection parameters if they're all set
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")
        
        if db_host and db_port and db_user and db_password and db_name:
            print(f"Connecting to database host: {db_host}:{db_port}")
            conn = psycopg2.connect(
                host = db_host,
                port = db_port,
                user = db_user,
                password = db_password,
                dbname = db_name,
                sslmode = sslmode,
                connect_timeout = connect_timeout,
            )
        else:
            db_url = os.getenv("SUPABASE_DB_URL")
            if db_url:
                print("Connecting to database using SUPABASE_DB_URL")
                conn = psycopg2.connect(dsn=db_url, sslmode=sslmode, connect_timeout=connect_timeout)
            else:
                raise ValueError("Database connection parameters not fully configured")
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise e

