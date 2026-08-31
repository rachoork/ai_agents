from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
# from tavily import TavilyClient

# tavily = TavilyClient()


# @tool
# def search(query:str) -> str:
#     """
#     Tool searches over the Internet
#     Args:
#         query: The query to search over for
#     Returns: The search result is 
#     """
#     print(f"searching for {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """Scheme for a source used by an Agent"""

    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Scheme for a response from an Agent"""

    answer:str = Field(description="The answer to the query")
    sources:List[Source] = Field(default_factory= list, description="The sources used to answer the query")

llm = ChatOpenAI(model="gpt-5")
# llm = ChatOllama(model="llama3.1:latest")

# tools = [search]

tools = [TavilySearch()]

agent = create_agent(model = llm, tools = tools, response_format=AgentResponse)

def main():
    print("Hello from ai-agents!")
    result = agent.invoke({"messages":HumanMessage(content="I want to find recent 3 software job trending in bay area related to software, computer science on Linkedin")})
    print(result)

if __name__ == "__main__":
    main()


# Agent Loop 
#  User Query --> Thought (Decides which tool to call or whihc action to take next?? through LLM -> SystemMessage (Genera Informaiton + All Tools available)) ---> 
# LLMs decides whihc tools ot call is decided by using Function Calling  
# User Query --> Thought by LLM --> Action by Tool Calling --> Observation
# Observation --> Thought by LLM --> Action by Tool Calling --> Observation --> Answer
                    #  ^                                              |
                    #   |_____________________________________________|