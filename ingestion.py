import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,log_warning)


load_dotenv()

# configure ssl content to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    show_progress_bar = True,
    chunk_size = 50,
    retry_min_seconds = 10
)

vectorstore = PineconeVectorStore(
    index_name = "documentation-agent-index",
    embedding = embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 2, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """process documents in batches asynchronously"""
    log_header("Vector Store Indexing Process")
    log_info(f"Vector Store Indexing Process with {len(documents)} documents", Colors.DARKCYAN)

    # create batches
    batches = [
        documents[i : i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(f"Vector Store Indexing Process with {len(batches)} batches", Colors.DARKCYAN)

    # process  all batches concurrently
    async def add_batch(batch: List[Document], batch_num:int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(f"Batch {batch_num} Indexed Successfully")
        except Exception as e:
            log_error(f"Error indexing Batch {batch_num}: {e}")
            return False
        return True
    # process batches concurrently
    tasks = [add_batch(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    log_success("Finished Vector Store Indexing Process")

    # Count sucessful Batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches): 
        log_success(f"Successfully indexed {successful}/{len(batches)} batches")
    else:
        log_warning(f"Failed to index {len(batches) - successful} batches")
    


async def main():
    print("Main Async Orchestraiton process to Ingest entire process....")

    log_header("Documentation Agent Ingestion Process")
    log_info("Starting Documentation Agent Ingestion Process", Colors.PURPLE)
    log_info("Starting Tavily Crawl Process", Colors.PURPLE)

    res = tavily_crawl.invoke({
        "url": "https://langchain.com/docs/",
        "max_depth": 5,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })

    all_docs = res["results"]
    print("___RESULT___", all_docs)
    log_success(f"Finished Tavily Crawl Process with {len(all_docs)} documents")

    # format docs
    formatted_docs = [
        Document(
            page_content=doc.get("raw_content") or doc.get("content") or "",
            metadata={
                "source": doc.get("url", ""),
                "title": doc.get("title", "")
            }
        )
        for doc in all_docs
    ]

    # Split the documents into chunks
    log_header("Document Chnking Phase ....")
    log_info(
        f"Text Splitter: Processing {len(formatted_docs)} docuemtn with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(formatted_docs)
    log_success(f"Finished Document Chnking Process with {len(splitted_docs)} documents") 

    # LLM Chunking Strategies
    # Fixed Size Chunking
    # Sliding Window Approach - captures the context between the chunks
    # Recursive Splitting - Semantic Chunking

    # process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)
    log_header("PIPELINE COMPLETED")
    log_success("Documentation Agent Ingestion Process Completed Successfully")
    log_info("Summary", Colors.BOLD)
    # log_info(f"    .URLs mapped: {len(site_map['results'])} ")
    log_info(f"    .Documents Indexed: {len(all_docs)}")
    log_info(f".   .Chunks Indexed: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())
 