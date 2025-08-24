#!/usr/bin/env python3
"""
ChromaDB Manager - handles connection, inserts, search, and cleanup.
"""

import os
import sys
import toml

# === CONFIG ===
CONFIG_PATH = "/Users/manoj/coding/x_config/config.toml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
config = toml.load(CONFIG_PATH)

chroma_cfg = config.get("chromadb", {})

# Add library path - EXACTLY like chroma_loader.py
if "output_path" in config and "lib" in config["output_path"]:
    lib_path = config["output_path"]["lib"]
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

# Logger - EXACTLY like chroma_loader.py
try:
    from logger import ScriptLogger
except ImportError as e:
    print(f"ERROR: Failed to import logger: {e}", file=sys.stderr)
    sys.exit(1)

# Now import other dependencies
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChromaManager:
    def __init__(self):
        self.client = None
        self.collection = None

        # Setup logger using config - EXACTLY like chroma_loader.py
        self.script_name = "chroma_manager"
        logs_path = config["output_path"]["logs"]
        self.logger = ScriptLogger(self.script_name, logs_path).initialize()

    def is_connected(self) -> bool:
        """Check if connected to ChromaDB."""
        return self.collection is not None

    def connect_to_chromadb(self) -> bool:
        """Connect to ChromaDB using config.toml values."""
        try:
            persist_directory = chroma_cfg.get("persist_directory")
            collection_name = chroma_cfg.get("collection", "documents")
            embedding_model = chroma_cfg.get("embedding_model", "all-MiniLM-L6-v2")
            distance_metric = chroma_cfg.get("distance_metric", "cosine")

            if not persist_directory:
                raise ValueError("Missing persist_directory in [chromadb] section of config.toml")

            self.logger.log_info(f"Connecting to ChromaDB at {persist_directory}")
            self.logger.log_info(f"Using embedding model: {embedding_model}, distance metric: {distance_metric}")

            # Init client
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Init embedding function
            embedding_func = SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_func,
                metadata={"hnsw:space": distance_metric}
            )

            self.logger.log_info(f"Connected to ChromaDB collection '{collection_name}'")
            return True

        except Exception as e:
            self.logger.log_error("Failed to connect to ChromaDB", e)
            return False

    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """Insert a document into the collection."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ChromaDB")

        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata or {}]
        )
        self.logger.log_info(f"Document added with ID {doc_id}")

    def search_documents(self, query: str, n_results: int = 3) -> dict:
        """Search documents in ChromaDB."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ChromaDB")

        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]  # Include metadata in results
        )

    def debug_search(self, query: str, n_results: int = 3) -> dict:
        """Search documents with debug logging."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ChromaDB")

        self.logger.log_info(f"Searching for: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]  # Include metadata in results
        )
        
        # Log the results for debugging
        if results and results['documents']:
            self.logger.log_info(f"Search returned {len(results['documents'][0])} results")
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                filename = metadata.get('filename', 'Unknown')
                self.logger.log_info(f"Result {i+1}: {filename} (distance: {results['distances'][0][i]:.4f})")
                self.logger.log_info(f"Preview: {doc[:100]}...")
        else:
            self.logger.log_info("Search returned 0 results")
            
        return results

    def get_all_documents(self) -> dict:
        """Fetch all documents from ChromaDB."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ChromaDB")

        return self.collection.get(
            include=["documents", "metadatas"]  # Include metadata when getting all docs
        )

    def disconnect(self):
        """Disconnect cleanly."""
        self.logger.log_info("Disconnecting ChromaDB client...")
        self.client = None
        self.collection = None