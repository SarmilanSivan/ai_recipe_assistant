"""
Defining all Nodes

"""

from langchain_core.messages import AIMessage, HumanMessage
import json

from agent.state import AgentState
from agent.chains import extraction_chain, sql_chain
from agent.schemas import ExtractionResult

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
        "mode_of_prep": result.mode_of_prep,
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