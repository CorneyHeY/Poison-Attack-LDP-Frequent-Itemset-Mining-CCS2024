import logging


class BasicMiner:
    skip_FO = False

    def __init__(self, user_dist, fo, func, loaded_dist=None, poison_input=None, **params):
        self.logger = logging.getLogger("basic_miner")
        self.logger.setLevel(logging.DEBUG)

        if poison_input is None:
            poison_input = []
        self.user_dist = user_dist
        self.loaded_dist = loaded_dist
        self.poison_input = poison_input
        self.est_dist = None
        self.FO = fo
        self.func = func
        self.func_params = params

    def find(self):
        if self.skip_FO:
            est_user_dist = self.user_dist
        elif self.loaded_dist is not None:
            est_user_dist = self.loaded_dist
        else:
            user_report = self.FO.perturb(self.user_dist)
            est_user_dist = self.FO.aggregate(user_report)
        est_poison_dist = self.FO.aggregate(self.poison_input)
        self.est_dist = est_user_dist + est_poison_dist
        return self.func(self.est_dist, **self.func_params)

