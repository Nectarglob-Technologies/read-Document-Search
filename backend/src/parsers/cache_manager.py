import json
import hashlib


class CacheManager:

    def __init__(self, cache_file="doc_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load()
        print(f"CacheManager initialized with {len(self.cache)} entries.")

    def _load(self):
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except:
            return {}

    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    '''
    def _hash_file(self, file_path):

        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    '''
        
    def _hash_file(self, file_path):
        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def get(self, file_path):

        file_hash = self._hash_file(file_path)

        return self.cache.get(file_hash)

    def set(self, file_path, parser_used):

        file_hash = self._hash_file(file_path)

        self.cache[file_hash] = parser_used
        self._save()
