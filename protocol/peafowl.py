import secrets
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.secret_sharing import share, reconstruct
from core.prf import PRF
from core.shprg import SHPRG
from core.permute_share import PermuteShare
from core.ot import OTExtension
from party.data_provider import DataProvider
from party.server import Server


class PEAFOWL:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_parties = config.get('num_parties', 3)
        self.num_samples = config.get('num_samples', 1024)
        self.num_features = config.get('num_features', 1024)
        self.modulus = config.get('secret_modulus', 2**64)
        self.precision_bits = config.get('precision_bits', 16)
        self.scale_factor = 2**self.precision_bits

    def encrypt_ids(self, data_provider: DataProvider) -> bytes:
        import pickle
        prf = PRF(data_provider.prf_key)
        encrypted_ids = []
        for uid in data_provider.get_ids():
            enc_id = prf.eval(uid.encode('utf-8'))
            encrypted_ids.append(enc_id)
        return pickle.dumps(encrypted_ids)

    def align_ids_offline(self, server: Server, encrypted_id_sets: Dict[str, bytes]) -> Tuple[Dict[str, List[int]], int]:
        import pickle
        id_lists = {}
        id_sets = {}
        for party_id, enc_data in encrypted_id_sets.items():
            id_lists[party_id] = pickle.loads(enc_data)
            id_sets[party_id] = set(id_lists[party_id])

        common_ids = set.intersection(*id_sets.values())
        intersection_size = len(common_ids)

        permutations = {}
        for party_id, enc_list in id_lists.items():
            common_list = sorted([x for x in enc_list if x in common_ids])
            permutations[party_id] = [enc_list.index(x) for x in common_list]

        return permutations, intersection_size

    def generate_seed_shares(self, data_provider: DataProvider, other_party_ids: List[str]) -> Dict[str, int]:
        seeds = {}
        for party_id in other_party_ids:
            seeds[party_id] = secrets.randbelow(self.modulus)
        return seeds

    def compute_feature_shares(self, data_provider: DataProvider, server_permutation: List[int]) -> List[int]:
        shprg = SHPRG(
            d=self.config.get('shprg_d', 8),
            m=data_provider.get_num_features(),
            q=self.config.get('shprg_q', 2**128),
            p=self.config.get('shprg_p', 2**64)
        )

        seeds = [secrets.randbelow(self.config.get('shprg_q', 2**128)) for _ in range(self.config.get('shprg_d', 8))]
        features = data_provider.get_features()

        aligned_size = min(len(server_permutation), len(features))
        aligned_features = np.zeros((aligned_size, features.shape[1]), dtype=np.float64)

        for i, perm_idx in enumerate(server_permutation[:aligned_size]):
            if perm_idx < len(features):
                aligned_features[i] = features[perm_idx]

        feature_shares = []
        for i in range(aligned_size):
            scaled_feature = aligned_features[i] * self.scale_factor
            int_feature = scaled_feature.astype(np.int64) % self.modulus
            share_i = share(int_feature.sum(), 1, self.modulus)[0]
            feature_shares.append(share_i)

        return feature_shares

    def permute_feature_shares(self, shares: List[int], permutation: List[int]) -> List[int]:
        permuter = PermuteShare(self.modulus)
        return permuter.permute_share_server(permutation, shares, None)

    def run_protocol(
        self,
        data_providers: List[DataProvider],
        server: Server
    ) -> Dict[str, np.ndarray]:
        encrypted_ids = {}
        for dp in data_providers:
            encrypted_ids[dp.get_id()] = self.encrypt_ids(dp)

        for party_id, enc_id in encrypted_ids.items():
            server.receive_encrypted_id(party_id, enc_id)

        permutations, intersection_size = self.align_ids_offline(server, encrypted_ids)

        aligned_features = {}
        for dp in data_providers:
            if dp.get_id() in permutations:
                perm = permutations[dp.get_id()]
                aligned_ids = [dp.get_ids()[i] for i in perm[:intersection_size]]
                dp.set_aligned_ids(aligned_ids)
                
                perm_array = np.array(perm[:intersection_size])
                features = dp.get_features()
                aligned = features[perm_array]
                aligned_features[dp.get_id()] = aligned

        return aligned_features

    def run_vertical_federated_learning(
        self,
        data_providers: List[DataProvider],
        server: Server
    ) -> Tuple[np.ndarray, np.ndarray]:
        aligned_features_dict = self.run_protocol(data_providers, server)
        
        X_combined = None
        for party_id in sorted(aligned_features_dict.keys()):
            if X_combined is None:
                X_combined = aligned_features_dict[party_id]
            else:
                X_combined = np.hstack([X_combined, aligned_features_dict[party_id]])
        
        y = None
        for dp in data_providers:
            if hasattr(dp, 'labels'):
                y = dp.labels
                break
        if y is None:
            aligned_ids = data_providers[0].get_aligned_ids()
            id_to_idx = {uid: idx for idx, uid in enumerate(data_providers[0].get_ids())}
            y = np.array([0] * len(aligned_ids))
        
        return X_combined, y
