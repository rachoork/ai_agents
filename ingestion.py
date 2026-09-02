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



if __name__ == "__main__":
    asyncio.run(main())
 