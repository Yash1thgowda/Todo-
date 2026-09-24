import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


tasks = []


@tool
def add_task(task: str) -> str:
    """Add a task to the to-do list."""
    task = task.strip()

    if not task:
        return "Task cannot be empty."

    tasks.append(task)
    return f"Task added: {task}"


@tool
def list_tasks() -> str:
    """List all tasks in the to-do list."""

    if not tasks:
        return "Your to-do list is empty."

    return "\n".join(
        f"{i + 1}. {task}"
        for i, task in enumerate(tasks)
    )


@tool
def remove_task(task: str) -> str:
    """Remove a task from the to-do list."""
    task = task.strip()

    for existing_task in tasks:
        if existing_task.lower() == task.lower():
            tasks.remove(existing_task)
            return f"Task removed: {existing_task}"

    return f"Task not found: {task}"


tools = [add_task, list_tasks, remove_task]


GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a simple To-Do List Agent. "
        "You help the user add, list, and remove tasks. "
        "Use the appropriate tool whenever the user wants to add, list, or remove a task. "
        "Keep responses short and clear."
    )
)


class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}


def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))

        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return str(content)

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)


app = FastAPI(
    title="To-Do Agent",
    version="1.0",
    description="A LangChain agent powered by Gemini for managing a simple to-do list."
)


@app.get("/")
def root():
    return {
        "message": "Server is running. Visit /agent/playground/ to use the To-Do Agent."
    }


add_routes(app, formatted_agent_chain, path="/agent")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)