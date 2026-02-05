from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple


class SharedAugmentationParams:
    """Shared-memory manager for augmentation parameters.

    Parameters
    ----------
    initial_params : dict[str, Any], optional
        Initial parameters to populate into shared memory.

    Notes
    -----
    This allows updating augmentation strength/schedules in the main process while
    dataloader workers read the latest values.
    """

    def __init__(self, initial_params: Optional[Dict[str, Any]] = None):
        self.manager = mp.Manager()
        self.params = self.manager.dict()
        self._lock = self.manager.Lock()
        self._version = self.manager.Value("i", 0)

        if initial_params:
            self.params.update(initial_params)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.params.get(key, default)

    def get_version(self) -> int:
        with self._lock:
            return int(self._version.value)

    def get_all_with_version(self) -> Tuple[Dict[str, Any], int]:
        with self._lock:
            return dict(self.params), int(self._version.value)

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.params)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.params[key] = value
            self._version.value += 1

    def update(self, params: Dict[str, Any]) -> None:
        with self._lock:
            self.params.update(params)
            self._version.value += 1

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self.params

    def __repr__(self) -> str:
        return f"SharedAugmentationParams({dict(self.params)})"
