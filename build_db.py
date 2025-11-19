import os
import re
import json
import shutil
from pathlib import Path

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup

def clean_html_content(content):
    """
    Uses BeautifulSoup to parse and clean HTML content from markdown.
    - Removes <img> and <details> tags completely.
    - Extracts text from other HTML tags.
    """
    content = re.sub(r'<img[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</?(p|ul|li|b|i|strong|em|br|span)[^>]*>', '', content)
    return content

def sanitize_filename(name):
    """Sanitizes a string to be a valid filename."""
    return re.sub(r'[^a-zA-Z0-9\u3131-\uD79D_.-]', '_', name)

def preprocess_file(file_path):
    """
    Reads a markdown file, cleans navigation headers and HTML tags,
    and returns a path to a temporary file with the cleaned content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Remove the navigation header from Confluence exports
        content = re.sub(r'^\\[.*?\\]\\(.*?\\\\.md\\)\\s*\\>\\s*', '', content, flags=re.MULTILINE)
        cleaned_content = clean_html_content(content)
        
        temp_file_path = file_path + ".tmp"
        with open(temp_file_path, 'w', encoding='utf-8') as temp_f:
            temp_f.write(cleaned_content)
        return temp_file_path
    except Exception as e:
        print(f"Error preprocessing file {file_path}: {e}")
        return None

def main():
    """
    Main function to build the vector databases based on topics defined in topics.json.
    """
    # --- Configuration ---
    topics_config_path = Path("rag_chatbot/topics.json")
    base_docs_path = Path("output")
    output_indexes_path = Path("faiss_indexes")

    # --- Load Topics ---
    if not topics_config_path.exists():
        print(f"Error: Configuration file not found at '{topics_config_path}'")
        return

    with open(topics_config_path, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    
    print(f"Found {len(topics)} topics to process in '{topics_config_path}'.")

    # --- Prepare Output Directory ---
    if output_indexes_path.exists():
        print(f"Clearing existing indexes in '{output_indexes_path}'...")
        shutil.rmtree(output_indexes_path)
    output_indexes_path.mkdir(parents=True, exist_ok=True)

    # --- Load Embedding Model ---
    print("Loading embedding model 'ko-sroberta-multitask'...")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded.")

    # --- Process Each Topic ---
    for topic in topics:
        topic_name = topic.get("name")
        topic_path_str = topic.get("path")

        if not topic_name or not topic_path_str:
            print(f"Skipping invalid topic entry: {topic}")
            continue

        print(f"\n--- Processing Topic: {topic_name} ---")

        # 1. Load Documents for the topic
        data_path = base_docs_path / topic_path_str
        if not data_path.exists():
            print(f"  [Warning] Document path not found, skipping: {data_path}")
            continue

        print(f"  Loading documents from: {data_path}")
        
        all_docs = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    temp_path = preprocess_file(file_path)
                    if temp_path:
                        loader = UnstructuredMarkdownLoader(temp_path, mode="single")
                        docs = loader.load()
                        # Add source metadata (relative to the topic's data_path)
                        for doc in docs:
                            doc.metadata['source'] = os.path.relpath(file_path, data_path)
                        all_docs.extend(docs)
                        os.remove(temp_path)

        if not all_docs:
            print("  No documents found for this topic. Skipping DB creation.")
            continue
            
        print(f"  Loaded {len(all_docs)} documents.")

        # 2. Split Documents
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )

        all_splits = []
        for doc in all_docs:
            splits = markdown_splitter.split_text(doc.page_content)
            for split in splits:
                split.metadata.update(doc.metadata)
            all_splits.extend(splits)

        print(f"  Split documents into {len(all_splits)} chunks.")

        # 3. Build and Save VectorStore
        if not all_splits:
            print("  No text chunks to process. Skipping DB creation.")
            continue

        print("  Creating and saving FAISS vector store...")
        try:
            vectorstore = FAISS.from_documents(all_splits, embeddings)
            
            # Sanitize topic name for directory creation
            sanitized_name = sanitize_filename(topic_name)
            save_path = output_indexes_path / sanitized_name
            save_path.mkdir(exist_ok=True)

            vectorstore.save_local(str(save_path))
            print(f"  FAISS index for '{topic_name}' saved successfully to '{save_path}'.")
        except Exception as e:
            print(f"  [Error] Failed to create or save vector store for '{topic_name}': {e}")

    print("\nAll topics processed.")

if __name__ == "__main__":
    main()
