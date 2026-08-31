from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
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



llm = ChatOpenAI(model="gpt-5")

# tools = [search]

tools = [TavilySearch()]

agent = create_agent(model = llm, tools = tools)

def main():
    print("Hello from ai-agents!")
    result = agent.invoke({"messages":HumanMessage(content="I want to find recent 3 software job trending in bay area related to software, computer science on Linkedin")})
    print(result)

if __name__ == "__main__":
    main()
