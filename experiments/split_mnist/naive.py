import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from models import MultiHeadMLP, MLP
from experiments.utils import set_seed, create_default_args
from collections import defaultdict
from avalanche.core import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict


class WeightAveraging(SupervisedPlugin):

    def __init__(self):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        self.state_dict = defaultdict(dict)

    def after_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter == 0:
            self.state_dict = strategy.model.state_dict()
        else:
            old_dict = self.state_dict
            # print('Old Dict')
            # print(old_dict)
            new_dict = strategy.model.state_dict()
            # print('New Dict')
            # print(new_dict)
            dicts = [old_dict, new_dict]
            num_models = len(dicts)
            for j, s_dict in enumerate(dicts):
                if j == 0:
                    uniform_soup = {k: v * (1. / num_models) for k, v in s_dict.items()}
                else:
                    uniform_soup = {k: v * (1. / num_models) + uniform_soup[k] for k, v in s_dict.items()}
            self.state_dict = uniform_soup
            # print('Uniform Soup')
            # print(uniform_soup)
            strategy.model.load_state_dict(uniform_soup)
            # print('New Params')
            # print(model.state_dict())


def naive_smnist(override_args=None):
    """
    "Continual Learning Through Synaptic Intelligence" by Zenke et. al. (2017).
    http://proceedings.mlr.press/v70/zenke17a.html
    """
    args = create_default_args({'cuda': 0, 'epochs': 10,
                                'learning_rate': 0.001, 'train_mb_size': 64, 'seed': 0,
                                'task_incremental': False}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                             args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=args.task_incremental,
                                          fixed_class_order=list(range(10)))

    model = MultiHeadMLP(hidden_size=256, hidden_layers=2) if args.task_incremental \
        else MLP(hidden_size=256, hidden_layers=2)

    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = avl.training.Naive(
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin, plugins=[WeightAveraging()])

    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = naive_smnist()
    print(res)
