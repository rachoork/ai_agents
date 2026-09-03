import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vectorstore
vectorstore = PineconeVectorStore(
    index_name="documentation-agent-index", embedding=embeddings
)

# Initialize chat models
model = init_chat_model("gpt-4o-mini", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer queries related to LangChain."""

    # Pass search_kwargs to as_retriever() instead of invoke()
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(
        query
    )

    # Fixed: Changed doc.metadata.get[...] to doc.metadata.get(...)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')} \n\n Content: {doc.page_content}"
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents as artifact
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline to answer questions using retrieved content.

    Args:
        query (str): The user's query.

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    system_prompt = (
        "You are a helpful AI Assistant that answers questions about LangChain documentation. "
        "You have access to a tool which retrieves relevant documentation. "
        "Use the tool to retrieve relevant documentation before answering the question. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the documentation, say that you don't know."
    )

    agent = create_agent(
        model, tools=[retrieve_context], system_prompt=system_prompt
    )

    messages = [{"role": "user", "content": query}]

    # Run the agent
    response = agent.invoke({"messages": messages})

    # Extract final response content
    answer = response["messages"][-1].content

    # Extract context documents from the ToolMessage artifact
    context_docs = []

    for message in response["messages"]:
        # Fixed: Changed message.artifacts to message.artifact (singular)
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if message.artifact:
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm("What are deep agents???")
    print("RESULT:----", result)