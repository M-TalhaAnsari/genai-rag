"""
Document parsing with Docling
DocumentProcessor class ensures efficient document parsing and retrieval 
Using ChromaDB for compatible chunking for vector search
A caching system to avoid redundant process
"""

# Required libraries
import os
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logging import logger

class DocumentProcessor:
    def __init__(self) -> None:
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded or local files."""
        total_size = 0
        for f in files:
            # FIX 1: Safely read file size depending on whether it's a path string or Streamlit object
            if isinstance(f, str):
                total_size += os.path.getsize(f)
            else:
                total_size += f.size  # Streamlit UploadedFile objects have a built-in .size attribute
                
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")
    
    def process(self, files: List) -> List:
        """Process files with caching for subsequent queries."""
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()
        
        for file in files:
            # Create a localized temporary tracking path for stream cleanup if needed
            created_temp_file = False
            
            try:
                # FIX 2: Handle both local file path strings and Streamlit uploads safely
                if isinstance(file, str):
                    file_name = os.path.basename(file)
                    file_path = file
                    with open(file, "rb") as f:
                        file_bytes = f.read()
                else:
                    file_name = file.name
                    file_bytes = file.getvalue()
                    
                    # Docling needs a physical file path to convert. We write the bytes to a temp file.
                    temp_dir = Path("temp_uploads")
                    temp_dir.mkdir(exist_ok=True)
                    file_path = str(temp_dir / file_name)
                    with open(file_path, "wb") as f:
                        f.write(file_bytes)
                    created_temp_file = True

                # Generate content-based hash for caching
                file_hash = self._generate_hash(file_bytes)
                cache_path = self.cache_dir / f"{file_hash}.pkl"
                
                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {file_name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {file_name}")
                    chunks = self._process_file(file_path, file_name)
                    # FIX 3: Changed 'self.cache_path' to local variable 'cache_path'
                    self._save_to_cache(chunks, cache_path)
                
                # Deduplicate chunks across files
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
                        
            except Exception as e:
                logger.error(f"Failed to process {file_name if 'file_name' in locals() else 'Unknown'}: {str(e)}")
                continue
            finally:
                # Clean up the temporary file if we generated one for an upload stream
                if created_temp_file and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    # FIX 4: Aligned the method name from '_process_files' to match your 'process()' loop invocation
    def _process_file(self, file_path: str, file_name: str) -> List:
        """Original processing logic with docling."""
        if not file_name.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
            logger.warning(f"Skipping unsupported file type: {file_name}")
            return []
            
        converter = DocumentConverter()
        markdown = converter.convert(file_path).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)
    
    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
    
    def _save_to_cache(self, chunks: List, cache_path: Path) -> None:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks
            }, f)

    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]
    
    # FIX 5: Completed the logical evaluation block for your cache expiration strategy
    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            cached_time = data.get("timestamp", 0)
            current_time = datetime.now().timestamp()
            
            # Check if the cache file age is less than our threshold setting
            expiration_seconds = settings.CACHE_EXPIRE_DAYS * 24 * 60 * 60
            return (current_time - cached_time) < expiration_seconds
        except Exception:
            return False