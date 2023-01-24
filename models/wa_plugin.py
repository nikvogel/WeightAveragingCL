from collections import defaultdict
from avalanche.core import SupervisedPlugin


class WeightAveragingPlugin(SupervisedPlugin):

    def __init__(self, alpha = 1):
        """Weight Averaging Plugin
        The plugin averages the weights of the old and retrained model after each training experience"""
        super().__init__()
        self.alpha = alpha
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
                uniform_soup[key] = self.alpha * (1 / 2 * (old_dict[key].add(new_dict[key]))) + \
                                    (1-self.alpha) * ((1 - strategy.clock.train_exp_counter ** (-1)) * old_dict[key] + \
                                        (strategy.clock.train_exp_counter ** (-1)) * new_dict[key])
            for key in new_keys:
                uniform_soup[key] = new_dict[key]
        
            self.state_dict = uniform_soup
            strategy.model.load_state_dict(uniform_soup)
