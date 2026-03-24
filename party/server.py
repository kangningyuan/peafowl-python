from typing import List, Dict, Any, Optional
from .base_party import BaseParty


class Server(BaseParty):
    def __init__(self, server_id: str, config: Dict[str, Any]):
        super().__init__(server_id, config)
        self.encrypted_ids = {}
        self.permutations = {}
        self.intersection_size = 0

    def receive_encrypted_id(self, party_id: str, encrypted_id: bytes):
        self.encrypted_ids[party_id] = encrypted_id

    def compute_intersection(self, encrypted_ids: Dict[str, bytes]) -> int:
        id_sets = {}
        for party_id, enc_id in encrypted_ids.items():
            import pickle
            id_sets[party_id] = set(pickle.loads(enc_id))

        if not id_sets:
            return 0

        intersection = set.intersection(*id_sets.values())
        self.intersection_size = len(intersection)
        return self.intersection_size

    def get_permutation(self, party_id: str) -> Optional[List[int]]:
        return self.permutations.get(party_id)

    def set_permutation(self, party_id: str, pi: List[int]):
        self.permutations[party_id] = pi

    def get_intersection_size(self) -> int:
        return self.intersection_size
