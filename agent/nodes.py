"""
Defining all Nodes

"""

from langchain_core.messages import AIMessage, HumanMessage
import json

from database.connection import execute_sql_query
from agent.state import AgentState
from agent.chains import extraction_chain, sql_chain, recommendation_chain, verifier_chain
from agent.schemas import ExtractionResult, VerificationResult

def extraction_node(state: AgentState):
    # Extract the current known parameters from the graph state
    current_params = state.get("structured_params", {})
    
    # Invoke the chain, passing the chat history and the current JSON state
    result: ExtractionResult = extraction_chain.invoke({
        "messages": state["messages"],
        "current_state": current_params
    })
    
    # Create an AI message object from the text the LLM generated
    ai_response_msg = AIMessage(content=result.ai_message)
    
    # Format the updated parameters back into a dictionary to save in state
    updated_params = {
        "ingredients": result.ingredients,
        "avoid_ingredients": result.avoid_ingredients,
        "max_calories": result.max_calories,
        "max_prep_time": result.max_prep_time,
        # "mode_of_prep": result.mode_of_prep,
        "is_sugar_free": result.is_sugar_free,
        "is_complete": result.is_complete
    }
    
    # Return the updates to LangGraph
    return {
        "messages": [ai_response_msg], # append this to the history
        "structured_params": updated_params,
        "extraction_status": "COMPLETE" if result.is_complete else "NEEDS_INFO"
    }


def human_node(state: AgentState):
    print("--- WAITING FOR USER INPUT ---")
    
    # Get the last message from the chat history - AI message
    last_message = state["messages"][-1]
    
    # Print the question to the console
    print(f"\nAssistant: {last_message.content}")
    
    # Pause the script and wait for the user to type their response
    user_text = input("You: ")
    
    # Wrap the text in a HumanMessage object, and return
    new_message = HumanMessage(content=user_text)
    
    return {"messages": [new_message]}


def sql_node(state: AgentState):
    print("--- GENERATING SQL QUERY ---")
    
    # Grab the completed JSON from the state
    params = state.get("structured_params", {})
    
    # Invoke the sql chain, convert the dict to a formatted JSON string for LLM
    raw_sql = sql_chain.invoke({
        "structured_params": json.dumps(params, indent=2)
    })
    
    # Clean up the output just in case the LLM ignored the markdown rule
    cleaned_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    print(f"Generated Query: {cleaned_sql}")
    
    # Update the state with the final query
    return {"sql_query": cleaned_sql}


def db_tool_node(state: AgentState):
    print("--- EXECUTING DATABASE QUERY ---")
    
    # Grab the generated query from the state
    query = state.get("sql_query", "")
    
    # Pass it to the tool
    results = execute_sql_query(query)
    
    # Update the LangGraph state with the exact database rows
    return {"db_results": results}


def recommendation_node(state: AgentState):
    print("--- DRAFTING RECOMMENDATION ---")
    
    constraints = state.get("structured_params", {})
    db_rows = state.get("db_results", [])
    
    # Check if the Verifier rejected the previous draft
    feedback = state.get("validation_feedback", "")
    
    if feedback:
        print(f"Applying QA Feedback to rewrite: {feedback}")
        # Format it clearly for the LLM
        feedback_context = f"Your previous draft was REJECTED for the following reason: '{feedback}'. Please rewrite your response to fix this exact issue."
    else:
        # First attempt, no feedback needed
        feedback_context = "No previous feedback. This is your first draft."

    # Invoke the LLM with the new feedback context
    draft = recommendation_chain.invoke({
        "user_constraints": json.dumps(constraints, indent=2),
        "database_results": json.dumps(db_rows, indent=2),
        "validation_feedback": feedback_context
    })
    
    # Return the new draft, and clear the feedback
    return {
        "draft_response": draft,
        "validation_feedback": "" 
    }

def verifier_node(state: AgentState):
    print("--- VERIFIER AGENT RUNNING ---")
    
    # Grab the necessary context from the state
    constraints = state.get("structured_params", {})
    db_rows = state.get("db_results", [])
    draft = state.get("draft_response", "")

    if hasattr(draft, "content"):
        draft = draft.content
    
    # Invoke the verifier
    result: VerificationResult = verifier_chain.invoke({
        "constraints": json.dumps(constraints),
        "db_results": json.dumps(db_rows),
        "draft": draft
    })
    
    if result.is_valid:
        print("Verification: PASS")
        # Turn the draft into an official AI message and add it to the chat history.
        final_message = AIMessage(content=draft)
        return {
            "verification_status": "PASS",
            "messages": [final_message]
        }
    else:
        print(f"Verification: FAIL - {result.feedback}")
        # Pass the feedback so the previous node knows how to fix it.
        return {
            "verification_status": "FAIL",
            "validation_feedback": result.feedback
        }