"""
Defining the promopts necessary for every the functions

"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a culinary extraction assistant. Your job is to gather meal requirements from the user to fill out a recipe search profile.

        CURRENT EXTRACTED DATA:
        {current_state}
        
        INSTRUCTIONS:
        1. Review the CURRENT EXTRACTED DATA. 
        2. Identify which core fields (ingredients, max_prep_time, mode_of_prep, max_calories, etc.) are still missing or null.
        3. If core fields are missing, set 'is_complete' to False and use the 'ai_message' field to ask the user a friendly, conversational question to gather ONE or TWO missing pieces of info. Do not overwhelm them with 5 questions at once.
        4. Update the fields based on the user's latest message. Retain previously gathered data.
        5. Once you have a sufficient profile (at least ingredients, time, and prep mode), set 'is_complete' to True and tell the user you are finding their recipe.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
])


sql_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a PostgreSQL expert. Your only job is to write a SQL query based on the user's extracted search parameters.

        DATABASE SCHEMA:
        CREATE TABLE recipes (
            id INT PRIMARY KEY,
            title VARCHAR(255),
            ingredients TEXT,
            calories INT,
            prep_time_mins INT,
            prep_mode VARCHAR(50),
            is_sugar_free BOOLEAN,
            url VARCHAR(500)
        );

        RULES:
        1. Return ONLY the raw SQL query. Do not wrap it in markdown formatting (like ```sql). Do not add any conversational text.
        2. Only use the columns listed in the schema.
        3. Always include 'LIMIT 3' to ensure we only return the top results.
        4. For ingredients, use the ILIKE operator to find partial matches in the text field. 
        Example: ingredients ILIKE '%chicken%' AND ingredients ILIKE '%spinach%'
        5. If a parameter in the JSON is null or empty, do not include it in the WHERE clause.

        """
        ),
        (
            "user",
            "Here are the extracted parameters: {structured_params}"
        )
])