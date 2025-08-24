#!/usr/bin/env python3
"""
RAG Engine - Retrieves context from ChromaDB and generates answers using LLM.
"""

import os
import sys
import toml
import traceback

# === CONFIG ===
CONFIG_PATH = "/Users/manoj/coding/x_config/config.toml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
config = toml.load(CONFIG_PATH)

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
from chroma_manager import ChromaManager
from llama_cpp import Llama


class RAGEngine:
    def __init__(self):
        self.config = config
        self.chroma = ChromaManager()
        self.llm = None

        # Setup logger using config - EXACTLY like chroma_loader.py
        self.script_name = "rag_engine"
        logs_path = self.config["output_path"]["logs"]
        self.logger = ScriptLogger(self.script_name, logs_path).initialize()

    def connect(self):
        if not self.chroma.connect_to_chromadb():
            self.logger.log_error("❌ Failed to connect to ChromaDB")
            return False

        try:
            llm_cfg = self.config["llm"]
            model_path = llm_cfg["model_path"]
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"LLM model not found: {model_path}")

            self.llm = Llama(
                model_path=model_path,
                n_ctx=llm_cfg.get("n_ctx", 4096),
                n_gpu_layers=llm_cfg.get("n_gpu_layers", 40),
                n_threads=llm_cfg.get("n_threads", 8),
                verbose=False,
            )

            self.logger.log_info(f"✅ Model loaded successfully from {model_path}")
            return True

        except Exception as e:
            self.logger.log_error(f"❌ Failed to load LLM: {e}")
            self.logger.log_error(traceback.format_exc())
            return False

    def build_prompt(self, query: str, contexts: list[str], sources: list[str]) -> str:
        """
        Build prompt with retrieved contexts and source information.
        """
        if not contexts:
            # If no relevant context is found, instruct the LLM to use its general knowledge
            # and not to mention any sources.
            context_text = "No relevant information found in the documents. Please answer using your general knowledge."
            source_text = "No sources available."
        else:
            context_text = "\n\n".join([f"Source: {sources[i]}\nContent: {contexts[i]}"
                                      for i in range(len(contexts))])
            source_text = ", ".join(set(sources))

        prompt = f"""<s>[INST] You are a helpful AI assistant. Use the following context to answer the question.
If the context is not relevant, answer the question using your general knowledge.
At the end of your response, only mention the source files if you used the provided context.

Context from documents:
{context_text}

Question: {query}

Please provide a helpful answer. [/INST]"""

        self.logger.log_info(f"Prompt built with {len(contexts)} context snippets from sources: {source_text}")
        return prompt






    def extract_filename_from_metadata(self, metadata: dict) -> str:
        """Extract filename from metadata."""
        if not metadata:
            return "Unknown source"
        
        possible_keys = ['filename', 'source', '_source_file']
        for key in possible_keys:
            if key in metadata:
                filename = metadata[key]
                return os.path.basename(str(filename))
        
        return "Unknown source"

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Perform retrieval-augmented generation and return both answer and sources.
        """
        try:
            if not self.chroma.is_connected():
                self.logger.log_error("Not connected to ChromaDB")
                return {"answer": "Error: Not connected to ChromaDB", "sources": []}

            self.logger.log_info(f"=== Question: {question} ===")
            results = self.chroma.debug_search(question, n_results=top_k)

            contexts, sources = [], []
            
            # --- START OF FIX: RELEVANCE FILTERING ---
            DISTANCE_THRESHOLD = 0.7  # For cosine distance, lower is better (0=exact match).

            if results and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0] # Get the distance scores

                self.logger.log_info(f"Found {len(documents)} initial results. Filtering with threshold {DISTANCE_THRESHOLD}...")

                for i, doc in enumerate(documents):
                    distance = distances[i]
                    if distance < DISTANCE_THRESHOLD:
                        self.logger.log_info(f"Accepting result {i+1} with distance {distance:.4f}")
                        contexts.append(doc[:1000])
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        sources.append(self.extract_filename_from_metadata(metadata))
                    else:
                        self.logger.log_warning(f"Discarding result {i+1} with distance {distance:.4f} (above threshold)")

                if contexts:
                    self.logger.log_info(f"Found {len(contexts)} relevant context snippets from sources: {sources}")
                else:
                    self.logger.log_warning("No relevant contexts found after filtering. The LLM will use general knowledge.")
            # --- END OF FIX ---
            else:
                self.logger.log_warning("No contexts found in initial search.")

            prompt = self.build_prompt(question, contexts, sources)

            if not self.llm:
                self.logger.log_error("LLM not loaded")
                return {"answer": "Error: LLM not loaded", "sources": sources}

            self.logger.log_info(f"Generating response for: '{question}'")
            response = self.llm(prompt, max_tokens=512, stop=["</s>", "[INST]"], echo=False)
            answer = response["choices"][0]["text"].strip()
            
            # Only return sources if they were actually used (i.e., contexts were found)
            final_sources = list(set(sources)) if contexts else []
            return {"answer": answer, "sources": final_sources}

        except Exception as e:
            self.logger.log_error(f"❌ Error in query processing: {e}\n{traceback.format_exc()}")
            return {"answer": f"Error processing your question: {e}", "sources": []}

    def generate_response(self, message: str) -> dict:
        """Wrapper for FastAPI chat endpoint."""
        try:
            if not self.llm:
                if not self.connect():
                    return {"error": "Failed to initialize LLM", "sources": []}
            result = self.query(message)
            return {"question": message, "answer": result["answer"], "sources": result["sources"]}
        except Exception as e:
            self.logger.log_error(f"❌ Error in generate_response: {e}\n{traceback.format_exc()}")
            return {"question": message, "answer": f"Error: {e}", "sources": []}

    def direct_response(self, message: str) -> str:
        """Generate response without using ChromaDB"""
        if not self.llm and not self.connect():
            return "Error: LLM not loaded"
        
        prompt = f"""<s>[INST] You are a helpful AI assistant. Answer the following question using your general knowledge.

Question: {message}
Answer: [/INST]"""
        
        self.logger.log_info(f"Generating direct response for: '{message}'")
        response = self.llm(prompt, max_tokens=512, stop=["</s>", "[INST]"], echo=False)
        return response["choices"][0]["text"].strip()