import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("Initializig components.....")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorStore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'], embedding=embeddings
    )

retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the following only based on following context
    {context}

    Question: {question}
    Provide detailed answer : 
    """
)

def format_docs(docs):
    """ Format retrieved documents into a single string"""
    return "\n\n".join([doc.page_content for doc in docs])
 

def create_retrieval_chain_with_llm():
    """
    Create a retrieval chain with LCEL (Langchain Expression Language)
    Returns a chain that can be invoked with {"question":"..."}
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving....")

    query = "Who is Ambareesh??"

    # =======================
    # Raw Implementation
    # =======================

    print("\n" + "=" * 70)
    print("IMPLEMENTATION: RAW LLM Implementiton (no RAG)")
    print("=" * 70 + "\n")

    result_raw = llm.invoke([HumanMessage(content=query)])

    print("\n Answer")

    print(result_raw.content)


# =======================
# RAG Implementation
# =======================

    print("\n" + "=" * 70)
    print("IMPLEMENTATION: RAG LLM Implementiton")
    print("=" * 70 + "\n")

    prompt = prompt_template.format(question=query, context=format_docs(retriever.invoke(query)))
    result = llm.invoke([HumanMessage(content=prompt)])

    print("\n Answer")

    print(result.content)

# RAG Implementaiton with LCEL (Better Approach)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION: RAG LLM Implementiton with LCEL")
    print("\n" + "=" * 70)

    chain_with_lcel = create_retrieval_chain_with_llm()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\n Answer")
    print(result_with_lcel)



