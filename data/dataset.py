import logging
import random
import heapq
from os import path
import pickle
import numpy as np
import sys


class DataSet:
    def __init__(self, config):
        self.data = None
        self.data_name = data_name = config["data_name"]
        self.test_percent = config["test_percent"]
        self.dict_size, self.user_total = self.get_params(data_name)

        self.load_data()

    def load_data(self):
        logging.info("Loading dataset %s-data..." % self.data_name)

        data_name = self.data_name
        file_dir = '%s-data' % data_name
        assert path.exists(file_dir), "data file path '%s' not exist" % file_dir

        file_path_prefix = '%s/%s' % (file_dir, data_name)
        overall_count = 0
        if not path.exists(file_path_prefix + '.pkl'):
            logging.debug("No existence of %s.pkl, loading data manually..." % file_path_prefix)
            data = []
            f = open(file_path_prefix + '.dat', 'r')
            for line in f:
                if len(line) == 0:
                    break
                if line[0] == '#':
                    continue
                data.append([])
                spliter = ',' if data_name == 'POS' else ' '
                items = line.split(spliter)
                for i in range(len(items)):
                    if data_name == 'POS' and i == len(items)-1:
                        continue
                    if data_name == 'IBM' and i < 2:
                        continue
                    item = int(items[i])
                    data[overall_count].append(item)
                data[overall_count].sort()
                overall_count += 1
                if overall_count >= self.user_total:
                    break
            pickle.dump(data, open(file_path_prefix + '.pkl', 'wb'))
        user_data = pickle.load(open(file_path_prefix + '.pkl', 'rb'))
        self.data = random.sample(user_data, int(len(user_data) * self.test_percent))

    def test_single(self, data):
        """ every user report a random item from all its items with no padding """
        results = np.zeros(self.dict_size, dtype=np.int)
        for i in range(len(data)):
            if len(data[i]) == 0:
                continue
            rand_index = np.random.randint(len(data[i]))
            value = data[i][rand_index]
            results[value] += 1

        return results

    def exp_single(self, data, knowledge=1):
        """ calculate the expectation of test_single distribution """
        results = np.zeros(self.dict_size, dtype=np.float)
        l = int(len(data) * knowledge)
        for i in range(l):
            length = len(data[i])
            for j in range(length):
                value = data[i][j]
                results[value] += 1.0 / knowledge / length
        return results

    def test_singleton_limit(self, data, length_limit):
        results = np.zeros(self.dict_size + 1, dtype=np.int)  # one extra place for dummy items
        for i in range(len(data)):
            values = data[i]
            # pad to L
            if len(values) > length_limit:
                rand_index = np.random.randint(len(values))
                result = values[rand_index]
            else:
                rand_index = np.random.randint(length_limit)
                if rand_index < len(values):
                    result = values[rand_index]
                else:
                    result = self.dict_size  # dummy item
            results[result] += 1
        return results[:-1]

    def exp_singleton_limit(self, data, length_limit, knowledge=1):
        results = np.zeros(self.dict_size, dtype=np.float)
        l = int(len(data) * knowledge)
        for i in range(l):
            values = data[i]
            # pad to L
            if len(values) > length_limit:
                for item in values:
                    results[item] += 1.0 / len(values)
            else:
                for item in values:
                    results[item] += 1.0 / length_limit

        results = results * (len(data) / l)
        return results

    def true_item_freq(self, data, k, knowledge=1):
        """ get true freq of top-k items """
        results = np.zeros(self.dict_size, dtype=np.int)
        l = int(len(data) * knowledge)
        for i in range(l):
            for j in range(len(data[i])):
                value = self.data[i][j]
                results[value] += 1
        results = results * (len(data) / l)
        return self.k_largest(results, k)

    @staticmethod
    def sample_from_dist(dist, percent):
        l1 = []
        for k, v in enumerate(dist):
            l1 += [k]*v
        l2 = random.sample(l1, int(len(l1)*percent))
        result = np.zeros(len(dist), dtype=np.float)
        for item in l2:
            result[item] += 1.0/percent
        return result.astype(np.int)

    @staticmethod
    def k_largest(results, top_k=None):
        if top_k is None:
            top_k = len(results)
        result = heapq.nlargest(top_k, range(len(results)), results.take)
        freq = np.zeros(top_k, dtype=np.int)
        for i in range(top_k):
            freq[i] = results[result[i]]
        return result, freq

    @staticmethod
    def test_length_cand(data, cand_list, limit, start_limit=0):
        """ every user report the length of its intersection with candidate set """
        results = np.zeros(limit - start_limit + 1, dtype=np.int)
        cand_set = set(cand_list)
        for i in range(len(data)):
            X = data[i]
            value = 0
            for item in X:
                if item in cand_set:
                    value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        return results

    @staticmethod
    def exp_length_cand(data, cand_list, limit, start_limit=0, knowledge=1):
        """ every user report the length of its intersection with candidate set """
        results = np.zeros(limit - start_limit + 1, dtype=np.float)
        cand_set = set(cand_list)
        l = int(len(data) * knowledge)
        for i in range(l):
            X = data[i]
            value = 0
            for item in X:
                if item in cand_set:
                    value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        results = results * (len(data) / l)
        return results

    @staticmethod
    def test_length(data, limit, start_limit=0):
        results = np.zeros(limit - start_limit + 1, dtype=np.int)
        for i in range(len(data)):
            value = len(data[i])
            results[value - start_limit] += 1
        return results

    @staticmethod
    def test_length_itemset(data, cand_dict, limit, start_limit=0):
        results = np.zeros(limit - start_limit + 1, dtype=np.int)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))
        for i in range(len(data)):
            current_set = singleton_set.intersection(set(data[i]))
            if len(current_set) == 0:
                continue
            value = 0
            for cand in cand_dict:
                if set(cand) <= current_set:
                    value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        return results

    @staticmethod
    def exp_length_itemset(data, cand_dict, limit, start_limit=0, knowledge=1):
        results = np.zeros(limit - start_limit + 1, dtype=np.int)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))
        l = int(len(data) * knowledge)
        for i in range(l):
            current_set = singleton_set.intersection(set(data[i]))
            if len(current_set) == 0:
                continue
            value = 0
            for cand in cand_dict:
                if set(cand) <= current_set:
                    value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        results = results * (len(data) / l)
        return results

    @staticmethod
    def test_singleton_cand_limit(data, key_dict: dict, length_limit):
        candidate_set = set(key_dict.keys())
        results = np.zeros(len(candidate_set) + 1, dtype=np.int)  # one extra place for dummy items
        for i in range(len(data)):
            values = []  # set of intersected items
            x = data[i]
            for item in x:
                if item in candidate_set:
                    values.append(item)
            # pad to L
            if len(values) > length_limit:
                rand_index = np.random.randint(len(values))
                result = key_dict[values[rand_index]]
            else:
                rand_index = np.random.randint(length_limit)
                if rand_index < len(values):
                    result = key_dict[values[rand_index]]
                else:
                    result = len(candidate_set)  # dummy item
            results[result] += 1
        return results[:-1]

    @staticmethod
    def exp_singleton_cand_limit(data, key_dict: dict, length_limit, knowledge=1):
        results = np.zeros(len(key_dict), dtype=np.float)
        l = int(len(data) * knowledge)
        for i in range(l):
            values = []  # set of intersected items
            x = data[i]
            for item in x:
                if item in key_dict.keys():
                    values.append(item)
            # pad to L
            if len(values) > length_limit:
                for item in values:
                    results[key_dict[item]] += 1.0 / len(values)
            else:
                for item in values:
                    results[key_dict[item]] += 1.0 / length_limit

        results = results * (len(data) / l)
        return results

    @staticmethod
    def test_cand_set_limit(data, cand_dict, length_limit):
        buckets = np.zeros(len(cand_dict) + 1, dtype=np.int)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))

        for i in range(len(data)):
            current_set = singleton_set.intersection(set(data[i]))
            if len(current_set) == 0:
                continue
            subset_count = 0
            subset_indices = []
            for cand in cand_dict:
                if set(cand) <= current_set:
                    subset_count += 1
                    subset_indices.append(cand_dict[cand])

            if subset_count > length_limit:
                rand_index = np.random.randint(subset_count)
                result = subset_indices[rand_index]
            else:
                rand_index = np.random.randint(length_limit)
                result = len(cand_dict)
                if rand_index < subset_count:
                    result = subset_indices[rand_index]

            buckets[result] += 1
        return buckets[:-1]

    @staticmethod
    def exp_cand_set_limit(data, cand_dict, length_limit, knowledge=1):
        buckets = np.zeros(len(cand_dict) + 1, dtype=np.float)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))

        l = int(len(data) * knowledge)
        for i in range(l):
            current_set = singleton_set.intersection(set(data[i]))
            if len(current_set) == 0:
                continue
            subset_count = 0
            subset_indices = []
            for cand in cand_dict:
                if set(cand) <= current_set:
                    subset_count += 1
                    subset_indices.append(cand_dict[cand])

            if subset_count > length_limit:
                for index in subset_indices:
                    buckets[index] += 1/subset_count
            else:
                for index in subset_indices:
                    buckets[index] += 1 / length_limit
        buckets = buckets * (len(data) / l)
        return buckets[:-1]

    @staticmethod
    def exp_cand_set(data, cand_dict, knowledge=1):
        buckets = np.zeros(len(cand_dict), dtype=np.int)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))

        length = int(len(data) * knowledge)
        for i in range(length):
            current_set = singleton_set.intersection(set(data[i]))
            if len(current_set) == 0:
                continue
            for cand in cand_dict:
                if set(cand) <= current_set:
                    index = cand_dict[cand]
                    buckets[index] += 1
        buckets = buckets * (len(data) / length)
        return buckets

    @staticmethod
    def get_params(data_name):
        if data_name == 'kosarak':
            dict_size = 42178
            user_total = 990002
        elif data_name == 'IBM':
            dict_size = 1000
            user_total = 1800000
        elif data_name == 'POS':
            dict_size = 1657
            user_total = 515596
        elif data_name == 'Foursquare':
            dict_size = 100191
            user_total = 2293
        elif data_name == 'Gowalla':
            dict_size = 1280969
            user_total = 107092
        elif data_name == 'MovieLens':
            dict_size = 1682
            user_total = 943
        else:
            assert False, "Unknown Dataset name '%s'" % data_name

        return dict_size, user_total

    @staticmethod
    def get_top_k(value_distribution, key_list, k):
        """ from top-k to top-1 """

        sorted_indices = np.argsort(value_distribution)
        sorted_indices = np.flip(sorted_indices)
        key_result = []
        value_result = []
        for j in sorted_indices[:k]:
            key_result.append(key_list[j])
            value_result.append(value_distribution[j])

        return key_result, value_result

    @staticmethod
    def get_percentile_cut(length_distribution, percent):
        summation = sum(length_distribution)
        current_sum = 0
        for i in range(1, len(length_distribution)):
            current_sum += length_distribution[i]
            if current_sum >= summation * percent:
                return i

        assert False, "shouldn't reach here"

    def true_itemset_freq(self, item_list, item_freq, top_k, knowledge=1):
        true_cand_set_map, true_cand_set_list = self.build_candidate_itemsets(item_list, item_freq, 4 * top_k)
        true_itemset_dist = self.exp_cand_set(self.data, true_cand_set_map, knowledge)
        true_itemset_list, true_itemset_freq_list = self.build_set_result(top_k, item_list, item_freq,
                                                                          true_cand_set_list, true_itemset_dist)
        return true_itemset_list, true_itemset_freq_list

    def build_candidate_itemsets(self, key_list, est_freq, k):
        cand_dict = {}
        for i in range(len(key_list)):
            cand_dict[key_list[i]] = est_freq[i]

        normalized_values = np.zeros(len(est_freq))
        for i in range(len(est_freq)):
            normalized_values[i] = (est_freq[i] * 0.9 / est_freq[0])
        cand_dict = {}
        cand_dict_prob = {}
        cand_set_list = []
        self.build_tuple_cand_bfs(cand_dict_prob, cand_dict, cand_set_list, key_list, normalized_values)
        cand_list = list(cand_dict.keys())
        cand_value = list(cand_dict_prob.values())
        sorted_indices = np.argsort(cand_value)
        cand_set_map = {}
        cand_set_list = []
        for j in sorted_indices[-k:]:
            cand_set_map[cand_list[j]] = len(cand_set_list)
            cand_set_list.append(tuple(cand_list[j]))
        return cand_set_map, cand_set_list

    @staticmethod
    def build_tuple_cand_bfs(cand_dict_prob, cand_dict, new_cand_inv, keys, values):
        ret = []
        cur = []
        for i in range(len(keys)):
            heapq.heappush(ret, (values[i], tuple([i])))
            heapq.heappush(cur, (-values[i], tuple([i])))
        while len(cur) > 0:
            new_cur = []
            while len(cur) > 0:
                (prob, t) = heapq.heappop(cur)
                for j in range(t[-1] + 1, len(keys)):
                    if len(ret) >= 3 * len(keys):
                        if -prob * values[j] > ret[0][0]:
                            heapq.heappop(ret)
                            l = list(t)
                            l.append(j)
                            heapq.heappush(ret, (-prob * values[j], tuple(l)))
                            heapq.heappush(new_cur, (prob * values[j], tuple(l)))
                    else:
                        l = list(t)
                        l.append(j)
                        heapq.heappush(ret, (-prob * values[j], tuple(l)))
                        heapq.heappush(new_cur, (prob * values[j], tuple(l)))
            cur = new_cur

        while len(ret) > 0:
            (prob, t) = heapq.heappop(ret)
            if len(t) == 1:
                continue
            l = list(t)
            new_l = []
            for i in l:
                new_l.append(keys[i])
            new_t = tuple(new_l)
            cand_dict[new_t] = len(new_cand_inv)
            cand_dict_prob[new_t] = prob
            new_cand_inv.append(new_t)

    @staticmethod
    def build_set_result(top_k, singletons_keys, singleton_freq, set_cand_dict_list, set_freq):
        current_estimates = np.concatenate((singleton_freq, set_freq), axis=0)
        itemset_list, itemset_freq_list = [], []
        sorted_indices = np.argsort(current_estimates)
        for j in sorted_indices[-top_k:]:
            if j < len(singletons_keys):
                itemset_list.append(tuple([singletons_keys[j]]))
                itemset_freq_list.append(singleton_freq[j])
            else:
                l = list(set_cand_dict_list[j - len(singletons_keys)])
                l.sort()
                itemset_list.append(tuple(l))
                itemset_freq_list.append(current_estimates[j])

        return itemset_list, itemset_freq_list

