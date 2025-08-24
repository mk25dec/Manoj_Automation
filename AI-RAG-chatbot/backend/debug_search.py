#!/usr/bin/env python3
"""
Quick script to check ChromaDB connection and run a test search.
"""

from chroma_manager import ChromaManager


def test_search():
    chroma = ChromaManager()
    print("=== TESTING CHROMADB SEARCH ===")

    try:
        # ✅ Connect first
        if not chroma.connect_to_chromadb():
            print("❌ Failed to connect to ChromaDB")
            return

        # ✅ Run debug search
        results = chroma.debug_search("wipe")

        print("\n=== SEARCH RESULTS ===")
        if results and results.get("documents") and results["documents"][0]:
            for i, (doc, metadata) in enumerate(
                zip(results["documents"][0], results["metadatas"][0])
            ):
                print(f"\n--- Result {i+1} ---")
                print(f"Content: {doc[:300]}...")
                print(f"Metadata: {metadata}")
        else:
            print("No results found!")

    except Exception as e:
        print(f"❌ Error during test search: {e}")

    finally:
        # ✅ Always disconnect
        chroma.disconnect()


if __name__ == "__main__":
    test_search()
