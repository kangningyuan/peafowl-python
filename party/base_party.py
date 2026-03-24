import secrets
from typing import List, Dict, Any, Optional
import numpy as np


class BaseParty:
    def __init__(self, party_id: str, config: Dict[str, Any]):
        self.party_id = party_id
        self.config = config
        self.shares = {}
        self.seeds = {}
        self.keys = {}
        self._init_keys()

    def _init_keys(self):
        key_bytes = self.config.get('prf_key_bytes', 16)
        self.prf_key = secrets.token_bytes(key_bytes)

    def get_id(self) -> str:
        return self.party_id

    def set_shares(self, key: str, value: Any):
        self.shares[key] = value

    def get_shares(self, key: str) -> Optional[Any]:
        return self.shares.get(key)

    def set_seeds(self, key: str, value: Any):
        self.seeds[key] = value

    def get_seeds(self, key: str) -> Optional[Any]:
        return self.seeds.get(key)

    def store_share(self, share_id: str, share_value: Any):
        self.shares[share_id] = share_value

    def retrieve_share(self, share_id: str) -> Optional[Any]:
        return self.shares.get(share_id)
