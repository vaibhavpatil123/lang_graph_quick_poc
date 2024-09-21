from typing import Annotated, List
from openai import ChatCompletion
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

from typing import List
from openai import ChatCompletion


def openai_llm(messages: List):
    formatted_messages = []

    # Ensure messages are non-empty
    if not messages:
        raise ValueError("The 'messages' list is empty.")

    # Format each message correctly for the OpenAI API
    for message in messages:
        if isinstance(message, dict):
            # Handle standard dict format (expected case)
            if "content" in message:
                role = message.get("role", "user")  # Default to 'user' if role is missing
                formatted_messages.append({"role": role, "content": message["content"]})
        elif hasattr(message, "content"):
            # Handle LangGraph message object format
            formatted_messages.append({"role": "user", "content": message.content})
        else:
            print(f"Warning: Skipping invalid message format: {message}")

    if not formatted_messages:
        raise ValueError("No valid messages found to send to OpenAI.")

    # Call the OpenAI API with the correctly formatted messages
    response = ChatCompletion.create(
        model="gpt-4",
        messages=formatted_messages
    )
    return response["choices"][0]["message"]["content"]


def chatbot(state: dict):
    if "messages" not in state or not state["messages"]:
        raise ValueError("No messages provided in the state.")

    # Pass the messages to the openai_llm function
    return {"messages": [openai_llm(state["messages"])]}


# Configure the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

# Optional: visualize the graph (requires extra dependencies)
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Start the interactive chatbot
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Ensure user input is formatted correctly
    message = [{"role": "user", "content": user_input}]

    # Stream the response from the graph
    for event in graph.stream({"messages": message}):
        for value in event.values():
            print("Assistant:", value["messages"][-1])
