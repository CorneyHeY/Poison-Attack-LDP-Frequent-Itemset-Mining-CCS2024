from protocols.FO import GRR, OLH, SH
from data.dataset import DataSet
import random
import numpy as np
import logging


class Attacker:
    def __init__(self, config, miner_config, dataset: DataSet):
        self.logger = logging.getLogger("Attacker")
        self.logger.setLevel(logging.DEBUG)

        self.config = config
        self.miner_config = miner_config
        self.mode = config["mode"]
        self.mima = config["mima"]
        self.dataset = dataset
        self.est_true_item_list, self.est_true_item_freq = \
            self.dataset.true_item_freq(self.dataset.data, miner_config["top_k"], self.config["knowledge"])
        self.est_true_itemset_list, self.est_true_itemset_freq = \
            self.dataset.true_itemset_freq(self.est_true_item_list, self.est_true_item_freq, miner_config["top_k"],
                                           self.config["knowledge"])

    def cand_select_attack(self, user_data, FO, freq_item_num, padding_length=None, user_dist=None):
        """ Attack phase 1 of SVIM """
        if self.mode == 0:
            return np.zeros(0, dtype=object), []

        atk_report_size = int(self.config["alpha"] * len(user_data))
        top_k = self.miner_config["top_k"]
        knowledge = self.config["knowledge"]

        if self.mima == 1:
            est_user_dist = self.dataset.sample_from_dist(user_dist, knowledge)
        else:
            if padding_length is None:
                est_user_dist = self.dataset.exp_single(user_data, knowledge)
            else:
                est_user_dist = self.dataset.exp_singleton_limit(user_data, padding_length, knowledge)
        expected_item_list, expected_freq_list = self.dataset.k_largest(est_user_dist)

    
        target_list = []
        bound_list = []
        atk_freq_list = []
        for i, item in enumerate(expected_item_list):
            if item in self.est_true_item_list:
                bound_list.append(expected_freq_list[i])
            else:
                target_list.append(item)
                atk_freq_list.append(expected_freq_list[i])

        target_list = target_list[0:freq_item_num] if self.mode >= 3 else random.sample(target_list, freq_item_num)
        target_freq_list = atk_freq_list[0:freq_item_num]

        result, atk_list = self.attack_resource_allocate(FO, atk_report_size, bound_list, target_list, target_freq_list)

        return result, atk_list

    def length_percentile_attack(self, user_data, olh, candidate_list, user_dist=None):
        """ Attack phase 2 of SVIM"""
        if self.mode < 4:
            return np.zeros(0, dtype=object)

        else:
            knowledge = self.config["knowledge"]
            length_limit = len(candidate_list)
            if self.mima == 1:
                est_user_dist = self.dataset.sample_from_dist(user_dist, knowledge)
            else:
                est_user_dist = self.dataset.exp_length_cand(user_data, candidate_list, length_limit,
                                                             knowledge=knowledge)
            assert type(olh) == OLH, "unsupported FO '%s'" % type(olh)
            atk_report_size = int(self.config["alpha"] * len(user_data))
            sample_time = self.config["sample_time"]
            valid_lengths = np.where(est_user_dist > 0)[0]
            upperbound = max(valid_lengths)
            atk_dist = np.zeros(length_limit + 1, dtype=int)
            atk_dist[length_limit] = atk_report_size
            results = olh.max_gain_perturb(atk_dist, range(upperbound + 1, length_limit + 1), sample_time)

            return results

    def finalist_select_attack(self, user_data, FO, freq_item_num, cand_map, padding_length, user_dist=None):
        """ Attack phase 3 of SVIM """
        if self.mode == 0:
            return np.zeros(0, dtype=object), []

        atk_report_size = int(self.config["alpha"] * len(user_data))
        epsilon = self.miner_config["epsilon"]
        top_k = self.miner_config["top_k"]
        knowledge = self.config["knowledge"]
        reverse_cand_map = dict(zip(cand_map.values(), cand_map.keys()))

        if self.mima == 1:
            est_user_dist = self.dataset.sample_from_dist(user_dist, knowledge)
        else:
            est_user_dist = self.dataset.exp_singleton_cand_limit(user_data, cand_map, padding_length, knowledge)
        expected_item_list, expected_freq_list = self.dataset.k_largest(est_user_dist)

        target_list = []
        bound_list = []
        atk_freq_list = []
        for i, key in enumerate(expected_item_list):
            item = reverse_cand_map[key]
            if item in self.est_true_item_list:
                bound_list.append(expected_freq_list[i])
            else:
                target_list.append(key)
                atk_freq_list.append(expected_freq_list[i])
        target_list = target_list[0:freq_item_num] if self.mode >= 3 else random.sample(target_list, freq_item_num)
        target_freq_list = atk_freq_list[0:freq_item_num]

        result, atk_list = self.attack_resource_allocate(FO, atk_report_size, bound_list, target_list, target_freq_list)
        atk_list = [reverse_cand_map[key] for key in atk_list]

        return result, atk_list

    def set_length_percentile_attack(self, user_data, olh, candidate_set_list, user_dist=None):
        """ Attack phase 2 of SVSM """
        if self.mode < 4:
            return np.zeros(0, dtype=object)

        else:
            knowledge = self.config["knowledge"]
            length_limit = len(candidate_set_list)
            if self.mima == 1:
                est_user_dist = self.dataset.sample_from_dist(user_dist, knowledge)
            else:
                est_user_dist = self.dataset.exp_length_itemset(user_data, candidate_set_list, length_limit,
                                                                knowledge=knowledge)
            assert type(olh) == OLH, "unsupported FO '%s'" % type(olh)
            atk_report_size = int(self.config["alpha"] * len(user_data))
            sample_time = self.config["sample_time"]
            valid_lengths = np.where(est_user_dist > 0)[0]
            upperbound = max(valid_lengths)
            atk_dist = np.zeros(length_limit + 1, dtype=int)
            atk_dist[length_limit] = atk_report_size
            results = olh.max_gain_perturb(atk_dist, range(upperbound + 1, length_limit + 1), sample_time)

            return results

    def final_itemset_select_attack(self, user_data, FO, freq_item_num, cand_map, padding_length):
        """ Attack phase 3 of SVSM """
        if self.mode == 0:
            return np.zeros(0, dtype=object), []

        atk_report_size = int(self.config["alpha"] * len(user_data))
        epsilon = self.miner_config["epsilon"]
        top_k = self.miner_config["top_k"]
        knowledge = self.config["knowledge"]
        reverse_cand_map = dict(zip(cand_map.values(), cand_map.keys()))

        est_user_dist = self.dataset.exp_cand_set_limit(user_data, cand_map, padding_length, knowledge)
        expected_item_list, expected_freq_list = self.dataset.k_largest(est_user_dist)

        target_list = []
        bound_list = []
        atk_freq_list = []
        for i, key in enumerate(expected_item_list):
            item = reverse_cand_map[key]
            if item in self.est_true_itemset_list:
                bound_list.append(expected_freq_list[i])
            else:
                target_list.append(key)
                atk_freq_list.append(expected_freq_list[i])
        target_list = target_list[0:freq_item_num] if self.mode >= 3 else random.sample(target_list, freq_item_num)
        target_freq_list = atk_freq_list[0:freq_item_num]

        result, atk_list = self.attack_resource_allocate(FO, atk_report_size, bound_list, target_list, target_freq_list)
        atk_list = [reverse_cand_map[key] for key in atk_list]

        return result, atk_list

    def atk_size_estimation(self, bound_list, freq_list, report_size, FO):
        a = 1.0 / (FO.p - FO.q)
        b = report_size * FO.q / (FO.p - FO.q)

        bound_list = [(bound + b) / a for bound in bound_list]
        freq_list = [(freq + b) / a for freq in freq_list]

        dummy_size = len(freq_list) - len(bound_list)
        atk_size = dummy_size
        for i, bound in enumerate(bound_list):
            current_atk_size = dummy_size + i + 1
            freq_increment = report_size * FO.average_gain_estimate(self.config["sample_time"], current_atk_size)
            freq_require = bound * current_atk_size - np.sum(freq_list[0:current_atk_size])
            if freq_require * 1.2 < freq_increment:
                atk_size = current_atk_size

        return atk_size

    def attack_resource_allocate(self, FO, atk_report_size, bound_list, target_list, target_freq_list):
        if self.mode < 4:
            atk_list = target_list
            atk_data = np.tile(atk_list, (atk_report_size, 1))
            atk_dist = self.dataset.test_single(atk_data)
            if self.mode == 1:
                result = FO.perturb(atk_dist)
            elif self.mode == 2:
                result = FO.perturb(atk_dist, 0)
            elif self.mode == 3 or self.mode == 2.5:
                result = FO.max_gain_perturb(atk_dist, target_list, self.config["sample_time"])
            else:
                assert False, "Unknown atk_mode '%d'" % self.mode

        elif self.mode == 4:
            bound_list.reverse()
            atk_size = self.atk_size_estimation(bound_list, target_freq_list, atk_report_size, FO)
            bound = bound_list[atk_size - (len(target_freq_list) - len(bound_list)) - 1] if len(bound_list) > 0 else 0

            atk_list = target_list[0:atk_size]
            a = 1.0 / (FO.p - FO.q)
            b = atk_report_size * FO.q / (FO.p - FO.q)
            freq_gap = [(bound - freq + b) / a for freq in target_freq_list[0:atk_size]]
            result = FO.max_gain_allocate(atk_report_size, self.config["sample_time"], atk_list, freq_gap)
        else:
            assert False, "Unsupported attacker mode '%s'" % self.mode

        return result, atk_list
