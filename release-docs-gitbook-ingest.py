#!/usr/bin/env python3
import os

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

from dotenv import load_dotenv
from utils import BaseLogger

import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain_community.document_loaders import (
    GitbookLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from chains import (
    load_embedding_model,
    load_llm,
)


load_dotenv(".env")

chroma_collection = os.getenv("CHROMA_COLLECTION", "release-docs")
chroma_host = os.getenv("CHROMA_HOST", "localhost")
chroma_port = int(os.getenv("CHROMA_PORT", 8000))
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

embeddings, dimension = load_embedding_model(
    embedding_model_name,
    config={"ollama_base_url": ollama_base_url},
    logger=BaseLogger(),
)


# Load environment variables
gitbook_url = os.environ.get('GITBOOK_URL', 'https://docs.release.com')
chunk_size = 500
chunk_overlap = 50


# Initialize Chroma client
chroma_client = chromadb.HttpClient(
    host=chroma_host,
    port=chroma_port,
    ssl=False,
    headers=None,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# create vector database if it doesn't exist
chroma_client.get_or_create_collection(chroma_collection, metadata={"key": "value"})


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {gitbook_url}")

    loader = GitbookLoader(gitbook_url, load_all_paths=True)
    documents = loader.load()

    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist() -> bool:
    """
    Checks if vectorstore exists
    """
    chroma_client.get_or_create_collection(chroma_collection, metadata={"key": "value"})
    return True

def main():

    does_vectorstore_exist()

    # Update and store locally vectorstore
    print(f"Appending to existing vectorstore")
    db = Chroma(client=chroma_client, collection_name=chroma_collection, embedding_function=embeddings)
    collection = db.get()

    texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
    print(f"Creating embeddings. May take some minutes...")
    db.add_documents(texts)

    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
