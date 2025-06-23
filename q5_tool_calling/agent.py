# agent.py

import os
from getpass import getpass
from langchain_core.tools import Tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

# === Define Tools ===

# 1. Python executor (runs a snippet)
@tool
def python_exec(code: str) -> str:
    """Executes Python code and returns the result."""
    try:
        # Create a dictionary to capture local variables
        local_vars = {}
        # Execute the code and capture its output
        exec(code, {}, local_vars)
        # Return the result if available
        if '_result' in local_vars:
            return str(local_vars['_result'])
        else:
            return "Code executed successfully, but no '_result' variable was defined."
    except Exception as e:
        return f"Error executing code: {str(e)}"

# 2. No-op tool (used when LLM wants to "think" but not run code)
@tool
def noop_tool(_input: str) -> str:
    """Returns nothing. Use this when you want to respond directly without executing code."""
    return ""

# === Create the Agent ===
llm = ChatOpenAI(temperature=0, model="gpt-4")

tools = [
    Tool.from_function(python_exec, name="python_exec", description="Executes Python code and returns the result."),
    Tool.from_function(noop_tool, name="noop", description="Use this to respond without running code."),
]

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the tools available to answer the user's question."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent using the new API
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

queries = [
    "How many 'r' in 'strawberry'?",
    "How many 'a' in 'banana'?",
    "How many 's' in 'Mississippi'?",
    "What is 12 * (3 + 2)?",
    "What is (2**5) - (10 / 2)?",
]

for q in queries:
    print(f"\nUser: {q}")
    response = agent_executor.invoke({"input": q})
    print(f"Agent: {response['output']}")