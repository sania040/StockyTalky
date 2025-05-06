import pandas as pd
from src.db.get_connection import get_db_connection

def execute_query(query, params=None, return_df=True):
    """
    Executes a database query and returns the results.
    
    Args:
        query (str): SQL query to execute
        params (tuple or dict, optional): Parameters for the query
        return_df (bool): If True, returns results as pandas DataFrame
                         If False, returns cursor and connection for manual handling
                         
    Returns:
        pd.DataFrame or tuple: Query results as DataFrame or (cursor, connection) tuple
    """
    conn = get_db_connection()
    
    try:
        if return_df:
            # Return results as DataFrame
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return df
        else:
            # Return cursor and connection for manual handling
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor, conn
    except Exception as e:
        print(f"Database query error: {e}")
        if conn and not conn.closed:
            conn.close()
        raise e

def execute_and_commit(query, params=None):
    """
    Executes an INSERT, UPDATE, or DELETE query and commits the changes.
    
    Args:
        query (str): SQL query to execute
        params (tuple or dict, optional): Parameters for the query
        
    Returns:
        bool: True if successful
    """
    cursor, conn = None, None
    
    try:
        cursor, conn = execute_query(query, params, return_df=False)
        conn.commit()
        return True
    except Exception as e:
        if conn and not conn.closed:
            conn.rollback()
        print(f"Database commit error: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn and not conn.closed:
            conn.close()