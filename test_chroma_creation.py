
import os
import chromadb
from chromadb.config import Settings

path = os.path.abspath("./cis_vector_db_test")
print(f"Testing Chroma creation at: {path}")

try:
    client = chromadb.PersistentClient(path=path)
    collection = client.create_collection(name="test_collection")
    collection.add(ids=["1"], documents=["test document"])
    print("Successfully created and added to collection.")
    client.clear_system_cache()
    # Try to delete it
    import shutil
    shutil.rmtree(path)
    print("Successfully cleaned up.")
except Exception as e:
    print(f"Error: {e}")
