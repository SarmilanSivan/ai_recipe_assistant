"""
Defining the Graph

"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal

from agent.state import AgentState
from agent.nodes import extraction_node, human_node, sql_node

def route_extraction(state: AgentState) -> Literal["sql_node", "human_node"]:
    """
    Inspects the state to determine if information gathering is complete.
    Returns the exact name of the next node to execute.
    """
    status = state.get("extraction_status")
    
    if status == "COMPLETE":
        print("--- ROUTING TO SQL GENERATOR ---")
        return "sql_node"
    
    print("--- ROUTING TO USER FOR MORE INFO ---")
    return "human_node"

# Initialize the graph with the State Class
workflow = StateGraph(AgentState)

# Add all Nodes
workflow.add_node("extraction_node", extraction_node)
workflow.add_node("human_node", human_node)
workflow.add_node("sql_node", sql_node)

# Add the standard edge to loop back from the user
workflow.add_edge("human_node", "extraction_node")

# Add the conditional edge to the extraction node
workflow.add_conditional_edges(
    "extraction_node",
    route_extraction, # The routing function
    {
        "sql_node": "sql_node",
        "human_node": "human_node"
    }
)

# Set the entry point
workflow.set_entry_point("extraction_node")

# Compile the graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_node"])