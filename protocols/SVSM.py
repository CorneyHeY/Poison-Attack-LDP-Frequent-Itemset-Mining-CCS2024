import logging
import math
import numpy as np
import os
from os import path
import heapq
from protocols.FO import GRR, OLH, BLH, SH
from protocols.basic_miner import BasicMiner
from data.dataset import DataSet
from data import aux_func
from adversary.attacker import Attacker
import pickle

class SVSM:
    phase1_percent = 0.4
    phase2_percent = 0.1
    phase3_percent = 0.5

    def __init__(self, config, dataset: DataSet, attacker: Attacker = None, load_id=None):
        self.name = "SVSM"
        self.logger = logging.getLogger("SVSM")
        self.logger.setLevel(logging.DEBUG)

        self.config = config
        self.top_k = config["top_k"]
        self.epsilon = config["epsilon"]
        self.dataset = dataset
        self.attacker = attacker

        self.loaded_dist = None
        self.load_id = load_id

        self.logger.info("config = %s" % self.config)

    def precompute(self):
        self.logger.info("SVIM precompute")
        num_users = int(len(self.dataset.data) * 0.5)
        p1, p2 = int(num_users * self.phase1_percent), int(num_users * (self.phase1_percent + self.phase2_percent))
        phase1_users = self.dataset.data[0: p1]
        dist = self.dataset.test_single(phase1_users)
        FO = OLH(self.dataset.dict_size, self.epsilon)
        output_dist = FO.fo(dist)
        if self.load_id is not None:
            file_dir = "precomputed_results"
            if not path.exists(file_dir):
                os.makedirs(file_dir)
            file_path_prefix = "%s/SVIM_%s_eps%s_%s_%s" % (file_dir, self.dataset.data_name, self.epsilon,
                                                           self.dataset.test_percent, self.load_id)
            pickle.dump(output_dist, open(file_path_prefix + '.pkl', 'wb'))

    def find(self):
        item_list, freq_list = self.svim_find()
        itemset_list, itemset_freq_list = self.svsm_find(item_list, freq_list)
        return itemset_list, itemset_freq_list

    def svim_find(self):
        num_users = int(len(self.dataset.data) * 0.5)

        p1, p2 = int(num_users * self.phase1_percent), int(num_users * (self.phase1_percent + self.phase2_percent))

        all_users = self.dataset.data[0: num_users]
        phase1_users = self.dataset.data[0: p1]
        phase2_users = self.dataset.data[p1: p2]
        phase3_users = self.dataset.data[p2: num_users]

        candidate_list, value_result = self.find_candidate_items(phase1_users)
        padding_length = self.find_padding_length(phase2_users, candidate_list)
        item_list, freq_list = self.find_top_k_items(phase3_users, candidate_list, padding_length)

        # utility test
        self.true_item_list, self.true_item_freq = \
            true_item_list, true_freq_list = self.dataset.true_item_freq(self.dataset.data, self.top_k)
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

    def svsm_find(self, item_list, freq_list):
        num_users = int(len(self.dataset.data) * 0.5)
        all_users = self.dataset.data[num_users:]
        users_cut = int(0.2 * len(all_users))
        phase4_users = all_users[0: users_cut]
        phase5_users = all_users[users_cut:]

        # step 1: build itemset candidate
        cand_set_map, cand_set_list = self.dataset.build_candidate_itemsets(item_list, freq_list, 2 * self.top_k)

        # step 2: itemset size distribution
        length_limit, set_length_distribution = self.find_set_padding_length(phase4_users, cand_set_list)

        # step 3: itemset est
        self.logger.info("Phase 5: find top-k itemsets")
        user_itemset_dist = self.dataset.test_cand_set_limit(phase5_users, cand_set_map, length_limit)
        use_grr, eps = self.set_grr(user_itemset_dist, length_limit)
        FO = GRR(len(user_itemset_dist), eps) if use_grr else OLH(len(user_itemset_dist), eps)
        perturbed_user_itemset_dist = FO.fo(user_itemset_dist)
        poison_input, atk_list = self.attacker.final_itemset_select_attack(phase5_users, FO,
                                                                      self.top_k, cand_set_map, length_limit)
        poison_itemset_dist = FO.aggregate(poison_input)
        est_itemset_dist = perturbed_user_itemset_dist + poison_itemset_dist
        est_itemset_dist *= 1 / 0.9 / 0.8

        # make up for dummy items
        self.update_tail_with_reporting_set(length_limit, set_length_distribution, est_itemset_dist)
        itemset_list, itemset_freq_list = self.dataset.build_set_result(self.top_k, item_list, freq_list,
                                                                        cand_set_list, est_itemset_dist)
        self.logger.info("Phase5 top-k itemsets = %s" % itemset_list)
        self.logger.debug("Phase5 attacked item list(%d) = %s" % (len(atk_list), atk_list))

        # utility test
        true_itemset_list, true_itemset_freq_list = self.dataset.true_itemset_freq(self.true_item_list,
                                                                                   self.true_item_freq, self.top_k)
        self.logger.info("true itemset list = %s" % true_itemset_list)
        sum_utilities = 0
        cand_num = 0
        est_num = 0
        for item in itemset_list:
            if item in true_itemset_list:
                cand_num += 1
        for i in range(self.top_k):
            if itemset_list[i] in true_itemset_list:
                est_num += 1
                index = true_itemset_list.index(itemset_list[i])
                sum_utilities += index + 1

        self.acc_itemset = accuracy2 = est_num / self.top_k
        self.ji_itemset = ji = est_num / (2 * self.top_k - est_num)
        self.ncr_itemset = ncr = sum_utilities / ((self.top_k + 1) * self.top_k / 2.0)
        self.logger.info("true_itemset_list = %s" % dict(zip(true_itemset_list, true_itemset_freq_list)))
        self.logger.info("itemset accuracy = %s" % accuracy2)
        self.logger.info("itemset ji = %s" % ji)
        self.logger.info("itemset ncr = %s" % ncr)
        return itemset_list, itemset_freq_list

    def find_candidate_items(self, users):
        if self.load_id is not None:
            file_dir = "precomputed_results"
            file_path_prefix = "%s/SVIM_%s_eps%s_%s_%s" % (file_dir, self.dataset.data_name, self.epsilon,
                                                           self.dataset.test_percent, self.load_id)
            if path.exists(file_path_prefix + '.pkl'):
                self.logger.info("Load precomputed distribution | eps=%s, id=%s" % (self.epsilon, self.load_id))
                self.loaded_dist = pickle.load(open(file_path_prefix + '.pkl', 'rb'))
            else:
                self.precompute()
                self.loaded_dist = pickle.load(open(file_path_prefix + '.pkl', 'rb'))

        self.logger.info("Phase 1: find candidate items")
        true_user_dist = self.dataset.test_single(users)
        FO1 = OLH(len(true_user_dist), self.epsilon)
        poison_input, atk_list = self.attacker.cand_select_attack(users, FO1, 2 * self.top_k, user_dist=true_user_dist)
        BM1 = BasicMiner(true_user_dist, FO1, self.dataset.get_top_k,
                         poison_input=poison_input, loaded_dist=self.loaded_dist,
                         key_list=range(self.dataset.dict_size), k=2 * self.top_k)
        candidate_list, value_result = BM1.find()

        # utility assessment
        self.logger.debug("Phase1 candidate list(%d) = %s" % (len(candidate_list), candidate_list))
        self.logger.debug("Phase1 attacked item list(%d) = %s" % (len(atk_list), atk_list))
        self.logger.debug("Phase1 candidate list = %s" % aux_func.check_list(candidate_list, atk_list))

        return candidate_list, value_result

    def find_padding_length(self, users, candidate_list):
        self.logger.info("Phase 2: decide padding length")
        user_length_dist = self.dataset.test_length_cand(users, candidate_list, len(candidate_list))
        FO2 = OLH(len(user_length_dist), self.epsilon)
        BM2 = BasicMiner(user_length_dist, FO2, self.dataset.get_percentile_cut,
                         poison_input=self.attacker.length_percentile_attack(users, FO2, candidate_list, user_length_dist),
                         percent=0.9)
        padding_length = BM2.find()
        self.logger.info("Phase2 padding length = %s" % padding_length)

        return padding_length

    def find_top_k_items(self, users, candidate_list, padding_length):
        self.logger.info("Phase 3: find top-k items")
        use_grr, eps = self.set_grr(candidate_list, padding_length)
        key_dict = dict(zip(candidate_list, range(len(candidate_list))))
        true_user_dist = self.dataset.test_singleton_cand_limit(users, key_dict, padding_length)
        FO3 = GRR(len(true_user_dist), eps) if use_grr else OLH(len(true_user_dist), eps)
        poison_input, atk_list = self.attacker.finalist_select_attack(users, FO3, self.top_k,
                                                                      key_dict, padding_length, true_user_dist)
        BM3 = BasicMiner(true_user_dist, FO3, self.dataset.get_top_k,
                         poison_input=poison_input,
                         key_list=candidate_list, k=self.top_k)
        item_list, value_result = BM3.find()
        self.logger.info("Phase3 top-k items = %s" % item_list)
        self.logger.debug("Phase3 attacked item list(%d) = %s" % (len(atk_list), atk_list))
        self.logger.debug("Phase3 candidate list = %s" % aux_func.check_list(item_list, atk_list))

        return item_list, value_result

    def find_set_padding_length(self, users, candidate_set_list):
        self.logger.info("Phase 4: decide itemset padding length")
        user_length_dist = self.dataset.test_length_itemset(users, candidate_set_list, len(candidate_set_list))
        FO4 = OLH(len(user_length_dist), self.epsilon)
        BM4 = BasicMiner(user_length_dist, FO4, self.dataset.get_percentile_cut,
                         poison_input=self.attacker.set_length_percentile_attack(users, FO4,
                                                                                 candidate_set_list, user_length_dist),
                         percent=0.9)
        padding_length = BM4.find()
        self.logger.info("Phase4 itemset padding length = %s" % padding_length)

        return padding_length, BM4.est_dist

    def find_top_k_itemsets(self, users, candidate_set_list, length_limit, set_length_distribution):
        self.logger.info("Phase 5: find top-k itemsets")
        use_grr, eps = self.set_grr(candidate_set_list, length_limit)
        key_dict = dict(zip(candidate_set_list, range(len(candidate_set_list))))
        true_user_dist = self.dataset.test_cand_set_limit(users, key_dict, length_limit)
        FO5 = GRR(len(true_user_dist), eps) if use_grr else OLH(len(true_user_dist), eps)
        poison_input, atk_list = self.attacker.finalist_select_attack(users, FO5, self.top_k, key_dict,
                                                                      length_limit, true_user_dist)
        BM5 = BasicMiner(true_user_dist, FO5, self.dataset.get_top_k,
                         poison_input=poison_input,
                         key_list=candidate_set_list, k=self.top_k)
        item_list, value_result = BM5.find()
        self.update_tail_with_reporting_set(length_limit, set_length_distribution, set_freq)
        self.logger.info("Phase5 top-k itemsets = %s" % item_list)
        self.logger.debug("Phase5 attacked item list(%d) = %s" % (len(atk_list), atk_list))
        self.logger.debug("Phase5 candidate list = %s" % aux_func.check_list(item_list, atk_list))

        return item_list, value_result

    def set_grr(self, new_cand_dict, length_limit):
        eps = self.epsilon
        use_grr = False
        if len(new_cand_dict) < length_limit * math.exp(self.epsilon) * (4 * length_limit - 1) + 1:
            eps = math.log(length_limit * (math.exp(self.epsilon) - 1) + 1)
            use_grr = True
        return use_grr, eps

    @staticmethod
    def update_tail_with_reporting_set(length_limit, length_distribution_set, value_result):
        addi_total_item = 0
        for i in range(length_limit + 1, len(length_distribution_set)):
            addi_total_item += length_distribution_set[i] * (i - length_limit)
            if length_distribution_set[i] == 0:
                break
        total_item = sum(value_result)

        ratio = addi_total_item * 1.0 / total_item
        for i in range(len(value_result)):
            value_result[i] *= (1.0 + ratio)
