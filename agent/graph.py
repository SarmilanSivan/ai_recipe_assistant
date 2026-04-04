"""
Defining the Graph

"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal

from agent.state import AgentState
from agent.nodes import (
    extraction_node,
    human_node,
    sql_node,
    db_tool_node,
    recommendation_node,
    verifier_node
)

# -- Define Conditional Routing Functions --

def route_after_extraction(state: AgentState) -> Literal["sql_node", "human_node"]:
    """
    Decides whether the info gathered is enough to query the database, 
    or need to ask the human for more details.
    """
    if state.get("extraction_status") == "COMPLETE":
        print("--- ROUTING TO SQL GENERATOR ---")
        return "sql_node"
    
    print("--- ROUTING TO USER FOR MORE INFO ---")
    return "human_node"

def route_after_verification(state: AgentState) -> Literal["__end__", "recommendation_node"]:
    """
    The QA Loop: If the verifier finds a hallucination, route back 
    to the recommendation node to rewrite it. Otherwise, finish.
    """
    if state.get("verification_status") == "PASS":
        return END
    
    # If it failed, loop back to fix the draft
    print("  [Router] QA Failed. Looping back to Recommendation Agent...")
    return "recommendation_node"

# -- Build the Graph --
# Initialize the graph with the State Class
workflow = StateGraph(AgentState)

# Add all Nodes to the graph
workflow.add_node("extraction_node", extraction_node)
workflow.add_node("human_node", human_node)
workflow.add_node("sql_node", sql_node)
workflow.add_node("db_tool_node", db_tool_node)
workflow.add_node("recommendation_node", recommendation_node)
workflow.add_node("verifier_node", verifier_node)

# -- Define the Edges --
# Start by extracting constraints from the user's message
workflow.add_edge(START, "extraction_node")

# Ask human or generate SQL
workflow.add_conditional_edges(
    "extraction_node",
    route_after_extraction, # The routing function
    {
        "sql_node": "sql_node",
        "human_node": "human_node"
    }
)

# Go back to extraction to update the JSON
workflow.add_edge("human_node", "extraction_node")

# The Database Pipeline
workflow.add_edge("sql_node", "db_tool_node")
workflow.add_edge("db_tool_node", "recommendation_node")

# Draft the response, then send it to QA
workflow.add_edge("recommendation_node", "verifier_node")

# Conditional: Did the draft pass QA?
workflow.add_conditional_edges(
    "verifier_node",
    route_after_verification,
    {
        END: END,
        "recommendation_node": "recommendation_node" # The Self-Correction Loop
    }
)

# Initialize memory to keep track of the chat history
memory = MemorySaver()

# Compile the graph, setting a breakpoint before the human_node runs
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_node"]
)