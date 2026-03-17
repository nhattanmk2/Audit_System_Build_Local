import os
import json
import hashlib
from pathlib import Path

CACHE_DIR = Path("./.llm_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(key_data: str) -> Path:
    hash_key = hashlib.md5(key_data.encode()).hexdigest()
    return CACHE_DIR / f"{hash_key}.json"

def get_cached_response(cache_key: str):
    cache_path = get_cache_path(cache_key)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cached_response(cache_key, data):
    cache_path = get_cache_path(cache_key)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def test_caching():
    test_key = "test_config_item_123"
    test_data = {"result": "success", "value": 42}
    
    # 1. Clear previous cache if exists
    path = get_cache_path(test_key)
    if path.exists():
        path.unlink()
        
    # 2. Test cache miss
    assert get_cached_response(test_key) is None
    print("Cache miss test passed")
    
    # 3. Test saving and hit
    save_cached_response(test_key, test_data)
    cached = get_cached_response(test_key)
    assert cached == test_data
    print("Cache hit test passed")
    
    # 4. cleanup
    path.unlink()
    print("Cleanup passed")

if __name__ == "__main__":
    try:
        test_caching()
        print("\nSUCCESS: Caching logic is working correctly.")
    except Exception as e:
        print(f"\nFAILURE: {e}")
