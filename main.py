# main.py

import os
import re
import warnings
from dotenv import load_dotenv

# LangChain components
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Logging
import logging
from logging_config import setup_logger

# Suppress pypdfium2 warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pypdfium2._helpers.textpage",
    message="get_text_range.*get_text_bounded",
)

# Setup logging
setup_logger("main_ingestion")
logger = logging.getLogger("main_ingestion")

# Load environment variables
load_dotenv()

# --- Global constants for paths ---
CHROMA_DB_PATH = "./chroma_db"
DATA_PATH = "./data"
PROCESSED_LOG_FILE = "./processed_files.log"

# ✅ UPDATED AND SAFER CLEANING FUNCTION ✅
def clean_pdf_text(text):
    """
    A more gentle cleaning function that removes common headers/footers
    without deleting important document content.
    """
    # Normalize whitespace and split into lines
    lines = text.split('\n')
    cleaned_lines = []

    # Patterns for lines to be REMOVED (e.g., page numbers, confidential watermarks)
    header_footer_patterns = [
        r'^\s*Page\s*\d+\s*(of\s*\d+)?\s*$',  # "Page 1", "Page 1 of 10"
        r'^\s*Confidential\s*$',
        r'^\s*DRAFT\s*$',
        r'(?i)government\s+of\s+telangana', # Case-insensitive match for the government header
    ]

    for line in lines:
        # Strip leading/trailing whitespace from the line
        stripped_line = line.strip()

        # Skip if the line is empty
        if not stripped_line:
            continue

        # Skip if the line matches any of the header/footer patterns
        if any(re.search(pattern, stripped_line, re.IGNORECASE) for pattern in header_footer_patterns):
            continue

        # Add the cleaned line to our list
        cleaned_lines.append(stripped_line)

    # Join the cleaned lines back into a single text block
    cleaned_text = " ".join(cleaned_lines)
    logger.info(f"Cleaned text: Retained {len(cleaned_lines)} lines.")
    return cleaned_text


def update_and_load_vector_db():
    """Process PDFs, clean text, add embeddings to ChromaDB, and log processed files."""
    logger.info("Starting database update process...")

    try:
        with open(PROCESSED_LOG_FILE, "r") as f:
            processed_files = set(f.read().splitlines())
    except FileNotFoundError:
        processed_files = set()
    logger.info(f"Found {len(processed_files)} previously processed file(s).")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        logger.info(f"Created data directory at: {DATA_PATH}")

    current_files = set(f for f in os.listdir(DATA_PATH) if f.endswith(".pdf"))
    logger.info(f"Found {len(current_files)} PDF(s) in the data directory: {', '.join(current_files)}")

    new_files_to_process = current_files - processed_files
    if not new_files_to_process:
        logger.info("Database is up-to-date. No new documents to add.")
        logger.info("Database update process finished.")
        return

    logger.info(f"Found {len(new_files_to_process)} new document(s) to process: {', '.join(new_files_to_process)}")

    all_chunks = []
    for file_name in new_files_to_process:
        file_path = os.path.join(DATA_PATH, file_name)
        logger.info(f"Processing '{file_name}'...")
        try:
            loader = PyPDFium2Loader(file_path)
            documents = loader.load()
            if not documents:
                logger.warning(f"No text extracted from '{file_name}'")
                continue
            
            # Apply the new, safer cleaning function to each document's content
            for doc in documents:
                doc.page_content = clean_pdf_text(doc.page_content)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)

            logger.info(f"Split '{file_name}' into {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to process '{file_name}': {e}")

    if all_chunks:
        logger.info("Initializing vector store and embedding model...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        )

        logger.info(f"Adding {len(all_chunks)} new chunks to the database...")
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            try:
                vector_store.add_documents(batch)
                logger.info(f"Added batch {i//batch_size + 1}/{-(-len(all_chunks)//batch_size)}")
            except Exception as e:
                logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")

        logger.info("Successfully added all new documents to the database.")

        with open(PROCESSED_LOG_FILE, "a") as f:
            for file_name in new_files_to_process:
                f.write(f"{file_name}\n")
        logger.info("Updated processed files log.")

    logger.info("Database update process finished.")

if __name__ == "__main__":
    update_and_load_vector_db()