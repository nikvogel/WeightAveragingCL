from collections import defaultdict
from avalanche.core import SupervisedPlugin


class WeightAveragingPlugin(SupervisedPlugin):

    def __init__(self):
        """Weight Averaging Plugin
        The plugin averages the weights of the old and retrained model after each training experience"""
        super().__init__()
        self.state_dict = defaultdict(dict)

    def after_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter == 0:
            self.state_dict = strategy.model.state_dict()
        else:
            old_dict = self.state_dict
            new_dict = strategy.model.state_dict()
            shared_keys = list(set(old_dict.keys()).intersection(new_dict.keys()))
            new_keys = list(set(new_dict.keys()).difference(old_dict.keys()))
            uniform_soup = defaultdict()
            for key in shared_keys:
                uniform_soup[key] = 1/2*(old_dict[key].add(new_dict[key]))
            for key in new_keys:
                uniform_soup[key] = new_dict[key]
            # for j, s_dict in enumerate(dicts):
            #     if j == 0:
            #         uniform_soup = {k: v * (1. / num_models) for k, v in s_dict.items()}
            #     else:
            #         uniform_soup = {k: v * (1. / num_models) + uniform_soup[k] for k, v in s_dict.items()}
            self.state_dict = uniform_soup
            strategy.model.load_state_dict(uniform_soup)