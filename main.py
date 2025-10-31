from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Define tools with decorator
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

@tool
def get_weather(location: str) -> str:
    """Get weather for a location"""
    return f"The weather in {location} is sunny and 72°F"

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Create agent
tools = [calculator, get_weather]
agent_executor = create_agent(llm, tools)

# Run the agent
response = agent_executor.invoke({
    "messages": [("user", "What's 25 * 4 and what's the weather in Paris?")]
})

print(response["messages"][-1].content)
