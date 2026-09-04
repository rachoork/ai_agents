from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()


@tool
def triple(num:float)-> float:
    """ Param is a number, return the triple of that number """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0).bind_tools(tools)

