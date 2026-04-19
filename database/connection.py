"""
Connections to the PostgreSQL database
"""

import os
import pandas as pd
from sqlalchemy.engine import URL
from dotenv import load_dotenv

from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

def get_engine():
    """Creates a SQLAlchemy engine for PostgreSQL"""
    
    # Create the URL object
    url_object = URL.create(
        "postgresql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME"),
    )
    
    try:
        engine = create_engine(url_object)
        return engine
    except Exception as e:
        print(f"Error creating engine: {e}")
        return None

def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """
    Connects to the PostgreSQL database via SQLAlchemy, executes the 
    LLM-generated query, and returns the results as a list of dictionaries.
    """
    print(f"\n[DB TOOL] Executing Query: {query}")
    
    if not query:
        return [{"error": "No query provided."}]
    
    engine = get_engine()

    try:
        sql = text(query)
        
        with engine.connect() as connection:
            result = connection.execute(sql)
            
            # .mappings() to convert Row objects into dict-like objects
            rows = result.mappings().fetchall()
            
            # Convert to standard Python dictionaries
            results = [dict(row) for row in rows]
            
        print(f"[DB TOOL] Success: {len(results)} rows found.")
        return results
        
    except SQLAlchemyError as e:
        # Catch SQL syntax errors, missing columns, or connection failures
        error_msg = f"SQL Execution Error: {str(e)}"
        print(f"\n[DB TOOL ERROR] {error_msg}")
        
        # Return the error inside the list so the LangGraph state receives
        return [{"error": error_msg}]
    
    except Exception as e:
        # Catch any other unexpected Python errors
        error_msg = f"Unexpected Error: {str(e)}"
        print(f"\n[DB TOOL ERROR] {error_msg}")
        return [{"error": error_msg}]


# Test the connection
if __name__ == "__main__":
    print("Testing PostgreSQL connection...")
    
    # A safe query to test if the table exists and data is readable
    #test_query = " SELECT name, header, link, ratings FROM recipe_body LIMIT 5;"
    test_query =""" SELECT name, link, tag
                    FROM recipe_body
                    WHERE 
                        'no added sugar' = ANY (SELECT LOWER(unnest(nutrition_profile)))
                        AND 'lemon juice' = ANY (SELECT LOWER(unnest(ingr_list)))
                        AND 'dijon mustard' != ALL (SELECT LOWER(unnest(ingr_list)))
                        AND calories < 400
                        AND carbs < 30; """
    
    test_results = execute_sql_query(test_query)
    
    print("\nTest Results:")
    for row in test_results:
        print(row)
