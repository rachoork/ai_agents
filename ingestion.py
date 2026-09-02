import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    print("Ingesting!....")
    print(os.environ['PINECONE_API_KEY'])
    print(os.environ['OPENAI_API_KEY'])
    loader = TextLoader("/Users/phalgunakumarrachoor/Desktop/study/ai_agents/mediumblog1.txt", encoding="UTF-8", autodetect_encoding=True)
    document = loader.load()

    print("Splitting .....")

    text_splitter = CharacterTextSplitter(chunk_size= 1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Split into {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

    print("Ingesting.....")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])

    print("Finish")
if __name__ == "__main__":
    main()
