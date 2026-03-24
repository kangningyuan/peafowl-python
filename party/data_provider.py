from typing import List, Dict, Any, Tuple
import numpy as np
from .base_party import BaseParty


class DataProvider(BaseParty):
    def __init__(self, provider_id: str, config: Dict[str, Any], ids: List[str], features: np.ndarray):
        super().__init__(provider_id, config)
        self.ids = ids
        self.features = features
        self.num_samples = len(ids)
        self.num_features = features.shape[1] if len(features.shape) > 1 else 1

    def get_ids(self) -> List[str]:
        return self.ids

    def get_features(self) -> np.ndarray:
        return self.features

    def get_num_samples(self) -> int:
        return self.num_samples

    def get_num_features(self) -> int:
        return self.num_features

    def get_id_bytes(self) -> List[bytes]:
        return [uid.encode('utf-8') for uid in self.ids]

    def set_aligned_ids(self, aligned_ids: List[str]):
        self.aligned_ids = aligned_ids

    def get_aligned_ids(self) -> List[str]:
        return getattr(self, 'aligned_ids', self.ids)

    def get_aligned_features(self) -> np.ndarray:
        aligned_ids = self.get_aligned_ids()
        id_to_idx = {uid: idx for idx, uid in enumerate(self.ids)}
        aligned_indices = [id_to_idx[uid] for uid in aligned_ids if uid in id_to_idx]
        return self.features[aligned_indices]
