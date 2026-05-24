"""
Persistent disk cache for official placement proxy scores.

Cache key: benchmark_name + placement_hash (8-char MD5 of positions rounded to 0.1 um)
Cache format: JSONL, one entry per line, appended on write.
Thread-unsafe; intended for single-process use per file.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class OfficialScoreCache:
    """Append-only JSONL cache for official placement proxy scores."""

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        disabled: bool = False,
        clear: bool = False,
    ):
        self._disabled = disabled or (cache_path is None)
        self._cache_path = Path(cache_path) if cache_path else None
        self._store: Dict[str, float] = {}
        self.hits: int = 0
        self.misses: int = 0

        if self._disabled:
            return

        if clear and self._cache_path is not None and self._cache_path.exists():
            self._cache_path.unlink()

        self._load()

    def _key(self, benchmark_name: str, placement_hash: str) -> str:
        return f"{benchmark_name}:{placement_hash}"

    def _load(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        k = self._key(entry["benchmark_name"], entry["placement_hash"])
                        if k not in self._store:
                            self._store[k] = float(entry["proxy_cost"])
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
        except OSError:
            pass

    def lookup(self, benchmark_name: str, placement_hash: str) -> Optional[float]:
        """Return cached proxy_cost, or None on miss. Updates hit/miss counters."""
        if self._disabled:
            return None
        k = self._key(benchmark_name, placement_hash)
        v = self._store.get(k)
        if v is not None:
            self.hits += 1
        else:
            self.misses += 1
        return v

    def record(
        self,
        benchmark_name: str,
        placement_hash: str,
        proxy_cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a new score entry. No-ops if disabled or entry already exists."""
        if self._disabled or self._cache_path is None:
            return
        k = self._key(benchmark_name, placement_hash)
        if k in self._store:
            return
        self._store[k] = proxy_cost
        entry: Dict[str, Any] = {
            "benchmark_name": benchmark_name,
            "placement_hash": placement_hash,
            "proxy_cost": proxy_cost,
            "timestamp": time.time(),
        }
        if metadata:
            entry["metadata"] = metadata
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def enabled(self) -> bool:
        return not self._disabled
