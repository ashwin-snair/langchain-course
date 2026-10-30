import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

_tavily = TavilySearch(max_results=3)

@tool
def search(query: str) -> str:
    """Search the web for job postings and information."""
    return str(_tavily.invoke({"query": query}))

llm = ChatOllama(model="llama3.1:8b", base_url=os.environ.get("OLLAMA_HOST"))
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer")]})
    print(result)

if __name__ == "__main__":
    main()