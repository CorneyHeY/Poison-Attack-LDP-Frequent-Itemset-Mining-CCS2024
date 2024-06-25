from data.dataset import DataSet
from protocols.SVSM import SVSM
from protocols.LDPMiner import LDPMiner
from adversary.attacker import Attacker
from data import aux_func
import random
import numpy as np

config = {
    "dataset_config": {
        "data_name": 'IBM',
        "test_percent": 1
    },
    "miner_config": {
        "top_k": 32,
        "epsilon": 4,
    },
    "attacker_config": {
        "mode": 0,  # 0: no attack,
        # 1: RRA
        # 2: RSA
        # 2.5: MGA-R
        # 3: MGA-Adv
        # 4: AOA
        "alpha": 0.01,  # malicious/benign
        "knowledge": 1,
        "mima": 0,  # attacker access to user reports/user data
        "sample_time": 1000
    }
}

dataset = DataSet(config["dataset_config"])
attacker = Attacker(config["attacker_config"], config["miner_config"], dataset)
miner = SVSM(config["miner_config"], dataset, attacker)
item_list, freq_list = miner.find()
