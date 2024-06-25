import logging
import math
import os
import pickle

import numpy as np
import heapq
from protocols.FO import GRR, OLH, BLH, SH, RAPPOR
from protocols.basic_miner import BasicMiner
from data.dataset import DataSet
from data import aux_func
from adversary.attacker import Attacker


class LDPMiner:
    def __init__(self, config, dataset: DataSet, attacker: Attacker = None, load_id=None):
        self.name = "LDPMiner"
        self.logger = logging.getLogger("LDPMiner")
        self.logger.setLevel(logging.DEBUG)

        self.config = config
        self.top_k = config["top_k"]
        self.epsilon = config["epsilon"]
        self.dataset = dataset
        self.attacker = attacker

        self.loaded_dist = None
        self.load_id = load_id

        self.logger.info("config = %s" % self.config)

    def precompute(self, users, padding_length):
        self.logger.info("LDPMiner precompute")
        dist = self.dataset.test_singleton_limit(users, padding_length)
        FO = BLH(len(dist), self.epsilon * 0.5)
        output_dist = FO.fo(dist)
        if self.load_id is not None:
            file_dir = "precomputed_results"
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_path_prefix = "%s/LDPMiner_%s_eps%s_%s_%s" % (file_dir, self.dataset.data_name, self.epsilon,
                                                           self.dataset.test_percent, self.load_id)
            pickle.dump(output_dist, open(file_path_prefix + '.pkl', 'wb'))

    def find(self):
        num_users = len(self.dataset.data)
        all_users = self.dataset.data[0: num_users]

        # for utility test
        true_item_list, true_freq_list = self.dataset.true_item_freq(all_users, self.top_k)

        padding_length = self.find_padding_length(all_users)
        candidate_list, value_result = self.find_candidate_items(all_users, padding_length)
        item_list, freq_list = self.find_top_k_items(all_users, candidate_list, padding_length)

        sum_utilities = 0
        cand_num = 0
        est_num = 0
        for item in candidate_list:
            if item in true_item_list:
                cand_num += 1
        for i in range(self.top_k):
            if item_list[i] in true_item_list:
                est_num += 1
                index = true_item_list.index(item_list[i])
                sum_utilities += index + 1
        accuracy1 = cand_num / self.top_k
        self.acc_item = accuracy2 = est_num / self.top_k
        self.ji_item = ji = est_num / (2 * self.top_k - est_num)
        self.ncr_item = ncr = sum_utilities / ((self.top_k + 1) * self.top_k / 2.0)
        self.logger.info("True top-k items = %s" % true_item_list)
        self.logger.info("candidate accuracy = %s" % accuracy1)
        self.logger.info("final accuracy = %s" % accuracy2)
        self.logger.info("ji = %s" % ji)
        self.logger.info("ncr = %s" % ncr)

        return item_list, freq_list

    def find_padding_length(self, users):
        length_dist = self.dataset.test_length(users, self.dataset.dict_size)
        padding_length = self.dataset.get_percentile_cut(length_dist, percent=0.9)
        return padding_length

    def find_candidate_items(self, users, padding_length):
        self.logger.info("Phase 1: find candidate items")
        if self.load_id is not None:
            file_dir = "precomputed_results"
            file_path_prefix = "%s/LDPMiner_%s_eps%s_%s_%s" % (file_dir, self.dataset.data_name, self.epsilon,
                                                               self.dataset.test_percent, self.load_id)
            if os.path.exists(file_path_prefix + '.pkl'):
                self.logger.info("Load precomputed distribution | eps=%s, id=%s" % (self.epsilon, self.load_id))
                self.loaded_dist = pickle.load(open(file_path_prefix + '.pkl', 'rb'))
            else:
                self.precompute(users, padding_length)
                self.loaded_dist = pickle.load(open(file_path_prefix + '.pkl', 'rb'))
        true_user_dist = self.dataset.test_singleton_limit(users, padding_length)
        FO1 = BLH(len(true_user_dist), self.epsilon * 0.5)
        poison_input, atk_list = self.attacker.cand_select_attack(users, FO1, 2 * self.top_k,
                                                                  padding_length=padding_length, user_dist=true_user_dist)
        BM1 = BasicMiner(true_user_dist, FO1, self.dataset.get_top_k,
                         poison_input=poison_input, loaded_dist=self.loaded_dist,
                         key_list=range(self.dataset.dict_size), k=2 * self.top_k)
        candidate_list, value_result = BM1.find()

        # utility assessment
        self.logger.info("Phase1 candidate list(%d) = %s" % (len(candidate_list), candidate_list))
        self.logger.info("Phase1 attacked item list(%d) = %s" % (len(atk_list), atk_list))
        self.logger.info("Phase1 candidate list = %s" % aux_func.check_list(candidate_list, atk_list))

        return candidate_list, value_result

    def find_top_k_items(self, users, candidate_list, padding_length):
        self.logger.info("Phase 2: find top-k items")
        key_dict = dict(zip(candidate_list, range(len(candidate_list))))
        true_user_dist = self.dataset.test_singleton_cand_limit(users, key_dict, padding_length)
        FO2 = RAPPOR(len(true_user_dist), self.epsilon * 0.5)
        poison_input, atk_list = self.attacker.finalist_select_attack(users, FO2, self.top_k,
                                                                      key_dict, padding_length, user_dist=true_user_dist)
        BM2 = BasicMiner(true_user_dist, FO2, self.dataset.get_top_k,
                         poison_input=poison_input,
                         key_list=candidate_list, k=self.top_k)
        item_list, value_result = BM2.find()
        self.logger.info("Phase2 top-k items = %s" % item_list)
        self.logger.debug("Phase2 attacked item list(%d) = %s" % (len(atk_list), atk_list))
        self.logger.debug("Phase2 candidate list = %s" % aux_func.check_list(item_list, atk_list))

        return item_list, value_result
