"""
Execution file

"""

import uuid
import warnings
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import app

warnings.filterwarnings("ignore")

def run_terminal_chat():
    print("========================================")
    print("🥣 Welcome to the Blue Recipe Assistant!")
    print("Type 'quit' or 'exit' to stop.")
    #Need to add assistant's capabilities for the user to understand
    print("========================================\n")

    # Create a unique thread ID for this terminal session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Get the initial message from the user
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        return

    # Setup the initial state
    current_input = {"messages": [HumanMessage(content=user_input)]}

    # The Execution Loop
    while True:
        # Stream the graph execution
        for event in app.stream(current_input, config, stream_mode="values"):
            # Grab the last message to print it
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                # Only print if it's an AI message
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    print(f"\nAssistant: {last_msg.content}")

        # Check the graph state to see if it reached the end or paused
        state = app.get_state(config)
        
        if not state.next:
            print("\n[Workflow Completed]")
            break
            
        if "human_node" in state.next:
            # Get the user's reply
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting...")
                break
                
            # Update the state with the user's input and act as the human_node
            app.update_state(
                config,
                {"messages": [HumanMessage(content=user_input)]},
                as_node="human_node"
            )
            
            # Set current_input to None so the graph resumes from the checkpointer's memory
            current_input = None 

if __name__ == "__main__":
    run_terminal_chat()