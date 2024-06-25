import numpy as np
import math
import xxhash
import time
import heapq

class FO:
    def fo(self, real_dist):
        assert False, "Missing overwrite of method 'fo'"

    def perturb(self, real_dist):
        assert False, "Missing overwrite of method 'perturb'"

    def max_gain_perturb(self, real_dist, targets, sample_time):
        assert False, "Missing overwrite of method 'fo'"

    def aggregate(self, noisy_samples):
        assert False, "Missing overwrite of method 'aggregate'"

    def average_gain_estimate(self, sample_time, targets):
        assert False, "Missing overwrite of method 'average_gain_estimate'"

    def max_gain_allocate(self, report_size, sample_time, atk_list, freq_gap):
        assert False, "Missing overwrite of method 'average_gain_estimate'"


class GRR(FO):
    def __init__(self, domain, eps):
        self.eps = eps
        self.domain = domain
        self.ee = np.exp(eps)
        self.p = self.ee / (self.ee + domain - 1)
        self.q = 1 / (self.ee + domain - 1)

    def average_gain_estimate(self, sample_time, targets):
        return 1

    def fo(self, real_dist):
        noisy_samples = self.perturb(real_dist)
        est_dist = self.aggregate(noisy_samples)
        return est_dist

    def perturb(self, real_dist, rand_on=1):
        print("rr perturbing...")
        p, domain = self.p, self.domain
        n = sum(real_dist)
        reports = np.zeros(n, dtype=np.int)
        counter = 0
        for k, v in enumerate(real_dist):
            for _ in range(v):
                y = x = k
                p_sample = np.random.random_sample()

                if rand_on and p_sample > p:
                    y = np.random.randint(0, domain - 1)
                    if y >= x:
                        y += 1
                reports[counter] = y
                counter += 1
        return reports

    def max_gain_perturb(self, real_dist, targets, sample_time):
        return self.perturb(real_dist, rand_on=0)

    def aggregate(self, noisy_samples):
        n = len(noisy_samples)
        p, q, domain = self.p, self.q, self.domain
        print("rr aggregating...")
        est = np.zeros(domain)
        unique, counts = np.unique(noisy_samples, return_counts=True)
        for i in range(len(unique)):
            est[unique[i]] = counts[i]

        a = 1.0 / (p - q)
        b = n * q / (p - q)
        est = a * est - b

        return est

    def max_gain_allocate(self, size, sample_time, atk_list, freq_gap):
        result = np.zeros(size, dtype=object)
        for i in range(size):
            index = 0
            # get max index
            for j, freq in enumerate(freq_gap):
                if freq > freq_gap[index]:
                    index = j

            result[i] = atk_list[index]
            freq_gap[index] -= 1

        return result

class OLH(FO):
    def __init__(self, domain, eps):
        self.g = int(np.exp(eps)) + 1
        if self.g < 3:
            self.g = 3
        self.p = 0.5
        self.q = 1 / self.g
        self.domain = domain
        self.eps = eps

    def fo(self, real_dist):
        domain = len(real_dist)
        noisy_samples = self.perturb(real_dist)
        est_dist = self.aggregate(noisy_samples)
        return est_dist

    def hash(self, item, sd):
        return xxhash.xxh32(str(int(item)), seed=sd).intdigest() % self.g

    def lh_maxgain_transfer(self, item, targets, sample_time):
        g, p, q = self.g, self.p, self.q
        seed = 0
        gain = 0
        # randomly sample to attain approximately max gain
        for _ in range(sample_time):
            rseed = np.random.randint(0, 100000)
            cur_gain = 0
            x = self.hash(item, rseed)
            # gain = num of targets supported by tuple (k,rseed)
            for t in targets:
                temp = self.hash(t, rseed)
                if temp == x:
                    cur_gain += 1
            if cur_gain > gain:
                gain = cur_gain
                seed = rseed
        y = self.hash(item, seed)
        return tuple([y, seed])

    def lh_maxgain_perturb1(self, real_dist, targets, sample_time):
        print("lh max-gain perturbing...")
        n = sum(real_dist)
        noisy_samples = np.zeros(n, dtype=object)
        counter = 0
        average_gain = 0.0
        for k, v in enumerate(real_dist):
            if v == 0:
                continue
            heap = [(0, -1)] * v
            # randomly sample to attain approximately max gain
            for _ in range(v*sample_time):
                rseed = np.random.randint(0, n)
                cur_gain = 0
                x = self.hash(k, rseed)
                # gain = num of targets supported by tuple (k,rseed)
                for t in targets:
                    temp = self.hash(t, rseed)
                    if temp == x:
                        cur_gain += 1
                heapq.heappushpop(heap, (cur_gain, rseed))

            for i in range(v):
                seed = heap[i][1]
                gain = heap[i][0]
                y = self.hash(k, seed)
                average_gain += gain
                noisy_samples[counter] = tuple([y, seed])
                counter += 1

        average_gain /= n
        print("average gain = ", average_gain)
        return noisy_samples

    def lh_maxgain_perturb2(self, real_dist, targets, sample_time):
        print("lh max-gain perturbing...")
        n = sum(real_dist)
        noisy_samples = np.zeros(n, dtype=object)
        counter = 0
        for k, v in enumerate(real_dist):
            for _ in range(v):
                seed = 0
                gain = 0
                # randomly sample to attain approximately max gain
                for _ in range(sample_time):
                    rseed = np.random.randint(0, n)
                    cur_gain = 0
                    x = self.hash(k, rseed)
                    # gain = num of targets supported by tuple (k,rseed)
                    for t in targets:
                        temp = self.hash(t, rseed)
                        if temp == x:
                            cur_gain += 1
                    if cur_gain > gain:
                        gain = cur_gain
                        seed = rseed
                # print("gain = ", gain)
                y = self.hash(k, seed)

                noisy_samples[counter] = tuple([y, seed])
                counter += 1
        return noisy_samples

    def max_gain_perturb(self, real_dist, targets, sample_time):
        return self.lh_maxgain_perturb1(real_dist, targets, sample_time)

    def average_gain_estimate(self, sample_time, targets):
        p, t = 1.0 / self.g, sample_time
        Pr = math.pow(1 - p, targets)
        F = Pr
        E = targets + 1 - math.pow(F, t)
        a = p / (1 - p)
        for i in range(targets):
            Pr = Pr * a * (targets - i) / (i + 1)
            F += Pr
            E -= math.pow(F, t)
        return E + 1

    def perturb(self, real_dist, rand_on=1):
        g, p, q = self.g, self.p, self.q
        print("lh perturbing...")
        n = sum(real_dist)
        noisy_samples = np.zeros(n, dtype=object)
        samples_one = np.random.random_sample(n)
        seeds = np.random.randint(0, n, n)

        counter = 0
        for k, v in enumerate(real_dist):
            for _ in range(v):
                y = x = self.hash(k, seeds[counter])
                # randomize the output at probability '1-p'
                if rand_on and samples_one[counter] > p:
                    y = np.random.randint(0, g - 1)
                    if y >= x:
                        y += 1
                noisy_samples[counter] = tuple([y, seeds[counter]])
                counter += 1
        return noisy_samples

    def aggregate(self, noisy_samples):
        g, p, q, domain = self.g, self.p, self.q, self.domain
        n = len(noisy_samples)
        print("lh aggregating...")
        print("len(noisy_samples) = ", n, "domain = ", domain)
        est = np.zeros(domain, dtype=np.int32)
        for i in range(n):
            # if n > 20 and i % (int(n / 20)) == 0:
            #     print("%.2f" % (i * 1.0 / n))
            for v in range(domain):
                x = self.hash(v, noisy_samples[i][1])
                if noisy_samples[i][0] == x:
                    est[v] += 1

        a = 1.0 / (p - q)
        b = n * q / (p - q)
        est = a * est - b
        return est

    def max_gain_allocate(self, size, sample_time, atk_list, freq_gap):
        result = np.zeros(size, dtype=object)
        for i in range(size):
            index = 0
            # get max index
            for j, freq in enumerate(freq_gap):
                if freq > freq_gap[index]:
                    index = j

            item = atk_list[index]
            tp = self.lh_maxgain_transfer(item, atk_list, sample_time)
            y, sd = tp[0], tp[1]
            assert (self.hash(item, sd) == y)
            # update gap
            for j, item in enumerate(atk_list):
                temp = self.hash(atk_list[j], sd)
                if temp == y:
                    freq_gap[j] -= 1
            result[i] = tp

        return result


class BLH(OLH):
    def __init__(self, domain, eps):
        super().__init__(domain, eps)
        ee = np.exp(eps)
        self.g = 2
        self.p = ee / (1+ee)
        self.q = 1.0 / (1+ee)
        self.domain = domain
        self.eps = eps

class SH(FO):
    # Sampling SH
    def __init__(self, domain, n, eps, seed=-1):
        self.eps = eps
        self.domain = domain  # 提交项值域
        self.n = n  # 用户数
        self.beta = 0.05  # 置信度beta
        self.ee = np.exp(eps)
        self.p = self.ee / (self.ee + 1)
        self.q = 1 / (self.ee + 1)
        self.m = int((np.log(self.domain + 1) * np.log(2 / self.beta) * (self.eps ** 2) * self.n) / np.log(
            np.e * self.domain / self.beta))
        print("SH m = ", self.m)
        self.seed = seed
        if seed == -1:
            self.seed = int(time.time())

    def hash(self, item):
        # print("item = ", item)
        return xxhash.xxh64(str(int(item)), seed=self.seed).intdigest()

    def rnd_proj(self, m, d):
        r = self.hash(m * self.domain + d) % 2
        r = r * 2 - 1  # {0,1} map into {-1,1}
        r = r / np.sqrt(self.m)
        return r

    def average_gain_estimate(self, sample_time, targets):
        p, q, t = 0.5, self.q, sample_time
        Pr = math.pow(1 - p, targets)
        F = Pr
        E = targets + 1 - math.pow(F, t)
        a = p / (1 - p)
        for i in range(targets):
            Pr = Pr * a * (targets - i) / (i + 1)
            F += Pr
            E -= math.pow(F, t)
        return E + 1

    def fo(self, real_dist):
        noisy_samples = self.sh_perturb(real_dist)
        est_dist = self.sh_aggregate(noisy_samples)
        return est_dist

    def perturb(self, real_dist, rand_on=0):
        print("sh perturbing...")
        ee, p, domain, m = self.ee, self.p, self.domain, self.m
        n = sum(real_dist)
        noisy_samples = np.zeros(n, dtype=object)
        samples_one = np.random.random_sample(n)

        counter = 0
        for k, v in enumerate(real_dist):
            for _ in range(v):
                r = np.random.randint(0, m)
                x = self.rnd_proj(r, k)
                y = (ee + 1) / (ee - 1) * m * x
                # randomize the output at probability '1-p'
                if rand_on and samples_one[counter] > p:
                    y = -y
                noisy_samples[counter] = tuple([r, y])
                counter += 1
        return noisy_samples

    def maxgain_transfer(self, item, targets, true_list, sample_time):
        p, q, m, ee = self.p, self.q, self.m, self.ee
        seed = 0
        gain = 0
        # randomly sample to attain approximately max gain
        for i in range(sample_time):
            rseed = np.random.randint(0, m)
            cur_gain = 0
            x = self.rnd_proj(rseed, item)
            # gain = num of targets supported by tuple (k,rseed)
            for t in targets:
                temp = self.rnd_proj(rseed, t)
                if temp == x:
                    cur_gain += 1
            for t in true_list:
                temp = self.rnd_proj(rseed, t)
                if temp == x:
                    cur_gain -= 1
            if cur_gain > gain:
                gain = cur_gain
                seed = rseed
        y = (ee + 1) / (ee - 1) * m * self.rnd_proj(seed, item)
        return tuple([seed, y])

    def maxgain_perturb(self, real_dist, atk_cand_list, true_list, sample_time):
        m, ee = self.m, self.ee
        print("sh max-gain perturbing...")
        n = sum(real_dist)
        noisy_samples = np.zeros(n, dtype=object)
        counter = 0
        for k, v in enumerate(real_dist):
            if v == 0: continue
            # print("k = %d, v = %d" % (k,v))
            index = 0
            gain = -100
            for _ in range(sample_time):
                rindex = np.random.randint(0, m)
                cur_gain = 0
                x = self.rnd_proj(rindex, k)
                for t in atk_cand_list:
                    temp = self.rnd_proj(rindex, t)
                    if temp == x:
                        cur_gain += 1
                for t in true_list:
                    temp = self.rnd_proj(rindex, t)
                    if temp == x:
                        cur_gain -= 1
                if cur_gain > gain:
                    gain = cur_gain
                    index = rindex

            for _ in range(v):
                x = self.rnd_proj(index, k)
                y = (ee + 1) / (ee - 1) * m * x
                noisy_samples[counter] = tuple([index, y])
                counter += 1
        return noisy_samples

    def aggregate(self, noisy_samples):
        p, q, domain = self.p, self.q, self.domain
        n = len(noisy_samples)
        print("sh aggregating...")
        print("len(noisy_samples) = ", n, "domain = ", domain)
        est = np.zeros(domain, dtype=np.float)
        for i in range(n):
            if n > 20 and (i + 1) % (int(n / 20)) == 0:
                print("%.2f" % ((i + 1) * 1.0 / n))
            r = noisy_samples[i][0]
            y = noisy_samples[i][1]
            for v in range(domain):
                est[v] += self.rnd_proj(r, v) * y

        a = 1.0 / (p - q)
        b = n * q / (p - q)
        # est = a * est - b
        return est


class RAPPOR(FO):
    """Basic RAPPOR"""
    def __init__(self, domain, eps):
        self.eps = eps
        self.domain = domain
        self.ee = np.exp(eps/2)
        self.p = self.ee / (self.ee + 1)
        self.q = 1 / (self.ee + 1)

    def average_gain_estimate(self, sample_time, num_targets):
        return num_targets

    def fo(self, real_dist):
        noisy_samples = self.perturb(real_dist)
        est_dist = self.aggregate(noisy_samples)
        return est_dist

    def perturb(self, real_dist, rand_on=1):
        print("rappor perturbing...")
        p, domain = self.p, self.domain
        n = sum(real_dist)
        reports = np.zeros(n, dtype=object)
        counter = 0
        for k, v in enumerate(real_dist):
            for _ in range(v):
                y = np.zeros(self.domain, int)
                y[k] = 1
                for i in range(domain):
                    p_sample = np.random.random_sample()
                    if rand_on and p_sample > p:
                        y[i] = 1 - y[i]

                reports[counter] = y
                counter += 1
        return reports

    def max_gain_perturb(self, real_dist, targets, sample_time):
        print("rappor perturbing...")
        p, domain = self.p, self.domain
        n = sum(real_dist)
        reports = np.zeros(n, dtype=object)
        counter = 0
        for k, v in enumerate(real_dist):
            for _ in range(v):
                y = np.zeros(self.domain, int)
                y[k] = 1
                for t in targets:
                    y[t] = 1

                reports[counter] = y
                counter += 1
        return reports

    def aggregate(self, noisy_samples):
        n = len(noisy_samples)
        p, q, domain = self.p, self.q, self.domain
        print("rappor aggregating...")
        est = np.sum(noisy_samples)

        a = 1.0 / (p - q)
        b = n * q / (p - q)
        est = a * est - b

        return est

    def max_gain_allocate(self, size, sample_time, atk_list, freq_gap):
        result = np.zeros(size, dtype=object)
        for i in range(size):
            y = np.zeros(self.domain, int)
            for index, t in enumerate(atk_list):
                y[t] = 1
                freq_gap[index] -= 1

            result[i] = y

        return result