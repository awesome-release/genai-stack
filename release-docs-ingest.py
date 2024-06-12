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
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
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
source_directory = os.environ.get('SOURCE_DIRECTORY', 'documents')
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



# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    # ".md": (UnstructuredMarkdownLoader, { "mode": "elements", "encoding": "utf8" }),
    ".md": (TextLoader, {"encoding": "utf8"}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    if os.path.getsize(file_path) != 0:
        filename, ext = os.path.splitext(file_path)
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            try:
                loader = loader_class(file_path, **loader_args)
                if loader:
                    return loader.load()
            except:
                print(f"Corrupted file {file_path}. Ignoring it.")
        else:
            print(f"Unsupported file {file_path}. Ignoring it.")
    else:
        print(f"Empty file {file_path}. Ignoring it.")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                if docs:
                    results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
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

    print(f"Ingestion complete! You can now query your documents")


if __name__ == "__main__":
    main()
