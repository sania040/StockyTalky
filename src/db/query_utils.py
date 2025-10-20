# src/db/query_utils.py

import pandas as pd
# REMOVED: No need to import get_db_connection here, as this file should not create connections.

def execute_query(conn, query, params=None):
    """
    Executes a SQL query and returns the result as a DataFrame.
    The connection is NOT closed by this function.
    """
    try:
        # This function should only handle returning a DataFrame.
        df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        print(f"Database query error: {e}")
        # Return an empty DataFrame on error to prevent the main script from crashing.
        return pd.DataFrame()

def execute_and_commit(conn, query, params=None):
    """
    Executes an INSERT, UPDATE, or DELETE query and commits the changes.
    The connection is NOT closed by this function.
    """
    cursor = None # Initialize cursor to None
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        print(f"Database commit error: {e}")
        # Roll back the transaction if anything goes wrong.
        if conn:
            conn.rollback()
        return False
    finally:
        # Always close the cursor, but never the connection.
        if cursor:
            cursor.close()