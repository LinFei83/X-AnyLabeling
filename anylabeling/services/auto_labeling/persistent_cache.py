"""Thread-safe persistent cache implementation for image embeddings."""

import os
import pickle
import hashlib
import threading
from pathlib import Path
import tempfile
import shutil


class PersistentCache:
    """Thread-safe persistent cache implementation for image embeddings."""

    def __init__(self, cache_dir=None, max_size_gb=50.0):
        """
        Initialize persistent cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, uses temp directory.
            max_size_gb: Maximum cache size in GB before cleanup.
        """
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "xanylabeling_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.lock = threading.Lock()
        
        # Create metadata file to track access times
        self.metadata_file = self.cache_dir / "metadata.pkl"
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Load metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return {}

    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception:
            pass

    def _get_cache_key(self, key):
        """Generate a safe filename from key."""
        # Use hash of the full path to create a unique filename
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return f"embedding_{key_hash}.pkl"

    def _get_cache_path(self, key):
        """Get full path for cache file."""
        return self.cache_dir / self._get_cache_key(key)

    def _cleanup_old_files(self):
        """Remove old cache files if size exceeds limit."""
        try:
            # Get all cache files with their sizes and access times
            cache_files = []
            total_size = 0
            
            for file_path in self.cache_dir.glob("embedding_*.pkl"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    total_size += size
                    
                    # Get access time from metadata, fallback to file modification time
                    access_time = self.metadata.get(file_path.name, file_path.stat().st_mtime)
                    cache_files.append((file_path, size, access_time))
            
            # If total size exceeds limit, remove oldest files
            if total_size > self.max_size_bytes:
                # Sort by access time (oldest first)
                cache_files.sort(key=lambda x: x[2])
                
                removed_size = 0
                target_size = self.max_size_bytes * 0.8  # Remove until 80% of limit
                
                for file_path, size, _ in cache_files:
                    if total_size - removed_size <= target_size:
                        break
                    
                    try:
                        file_path.unlink()
                        removed_size += size
                        # Remove from metadata
                        self.metadata.pop(file_path.name, None)
                    except Exception:
                        pass
                
                self._save_metadata()
                
        except Exception:
            pass

    def get(self, key):
        """Get value from cache. Returns None if key is not present."""
        with self.lock:
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                return None
            
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access time
                import time
                self.metadata[cache_path.name] = time.time()
                self._save_metadata()
                
                return value
            except Exception:
                # If file is corrupted, remove it
                try:
                    cache_path.unlink()
                    self.metadata.pop(cache_path.name, None)
                except Exception:
                    pass
                return None

    def put(self, key, value):
        """Put value into cache."""
        with self.lock:
            try:
                cache_path = self._get_cache_path(key)
                
                # Save to temporary file first, then move to avoid corruption
                temp_path = cache_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Atomic move
                shutil.move(str(temp_path), str(cache_path))
                
                # Update metadata
                import time
                self.metadata[cache_path.name] = time.time()
                self._save_metadata()
                
                # Cleanup if needed
                self._cleanup_old_files()
                
            except Exception:
                # Clean up temp file if it exists
                temp_path = self._get_cache_path(key).with_suffix('.tmp')
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass

    def find(self, key):
        """Returns True if key is in cache, False otherwise."""
        with self.lock:
            return self._get_cache_path(key).exists()

    def clear(self):
        """Clear all cache files."""
        with self.lock:
            try:
                for file_path in self.cache_dir.glob("embedding_*.pkl"):
                    file_path.unlink()
                self.metadata.clear()
                self._save_metadata()
            except Exception:
                pass

    def get_cache_info(self):
        """Get cache statistics."""
        with self.lock:
            try:
                cache_files = list(self.cache_dir.glob("embedding_*.pkl"))
                total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
                return {
                    'files': len(cache_files),
                    'size_mb': total_size / (1024 * 1024),
                    'size_gb': total_size / (1024 * 1024 * 1024),
                    'cache_dir': str(self.cache_dir)
                }
            except Exception:
                return {'files': 0, 'size_mb': 0, 'size_gb': 0, 'cache_dir': str(self.cache_dir)} 