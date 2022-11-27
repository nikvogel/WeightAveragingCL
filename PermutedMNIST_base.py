from collections import defaultdict
from avalanche.core import SupervisedPlugin
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, ExperienceForgetting, StreamConfusionMatrix, \
    cpu_usage_metrics, timing_metrics, ram_usage_metrics, gpu_usage_metrics, disk_usage_metrics, MAC_metrics, \
    forgetting_metrics, confusion_matrix_metrics
from avalanche.logging import TextLogger, InteractiveLogger, TensorboardLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin, EWCPlugin
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive


class WeightAveraging(SupervisedPlugin):

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
            dicts = [old_dict, new_dict]
            num_models = len(dicts)
            for j, s_dict in enumerate(dicts):
                if j == 0:
                    uniform_soup = {k: v * (1. / num_models) for k, v in s_dict.items()}
                else:
                    uniform_soup = {k: v * (1. / num_models) + uniform_soup[k] for k, v in s_dict.items()}
            self.state_dict = uniform_soup
            strategy.model.load_state_dict(uniform_soup)


benchmark = PermutedMNIST(n_experiences=10, seed=1)
model = SimpleMLP(num_classes=benchmark.n_classes)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = CrossEntropyLoss()

text_logger = TextLogger(open('log.txt', 'w'))
interactive_logger = InteractiveLogger()
tensorboard_logger = TensorboardLogger()
csv_logger = CSVLogger(log_folder='./csv_log*')

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=False,
                             stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tensorboard_logger, csv_logger]
)
ewc = EWCPlugin(ewc_lambda=0.001)

strategy = Naive(model=model, optimizer=optimizer, criterion=criterion, train_mb_size=128,
                 plugins=[WeightAveraging(), ewc], evaluator=eval_plugin)
strategy.train(benchmark.train_stream)
strategy.eval(benchmark.test_stream)
