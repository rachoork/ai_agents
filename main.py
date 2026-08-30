import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from ai-agents!")
    print(os.getenv("OPENAI_API_KEY"))

    information = """
    Elon Reeve Musk is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. 
    Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire in June 2026; 
    as of August 14, 2026, Forbes estimates his net worth to be US$864 billion
    """

    summary_tmplate = """
    given the information {information} about a person I wanted to you to create:
    1. A short summary of the person,
    2. Two interesting thigns abt that person,
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_tmplate)

    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model= "gemma3:270m")

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})

    print(response)



if __name__ == "__main__":
    main()

# langchain workflow
#
# User Query --> Prompt Template      -->    Language Model     -->     Output parser           -->  External APi Tool Call   -->   Final LLM Call          --> Final Output
#               (format the query                                       (Parse LLM output into       (Call Externla Services)       (Process API response)
#               into structured prompt)      (Generate Response)        structured data)
