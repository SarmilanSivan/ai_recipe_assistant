"""
Agent State definition

"""

from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Handle user, AI messages
    messages: Annotated[List[BaseMessage], add_messages]

    extraction_status: str
    
    # Holds the ongoing, partially filled recipe JSON dictionary
    structured_params: Dict[str, Any]
    
    # "COMPLETE" or "NEEDS_INFO" - will be used by the router
    extraction_status: str
    
    # Save the SQL query
    sql_query: str

    # need to refine
    db_results: List[Dict[str, Any]]
    draft_response: str
    validation_feedback: str
    verification_status: str

    revision_number: int