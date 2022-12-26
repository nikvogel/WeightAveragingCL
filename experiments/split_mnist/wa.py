import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.models import MTSimpleMLP, SimpleMLP
from avalanche.evaluation import metrics as metrics
from models import MultiHeadMLP, MLP, WeightAveragingPlugin
from experiments.utils import set_seed, create_default_args


def wa_smnist(override_args=None):
    """
    "Continual Learning Through Synaptic Intelligence" by Zenke et. al. (2017).
    http://proceedings.mlr.press/v70/zenke17a.html
    """
    args = create_default_args({'cuda': 1,
                                'epochs': 3,
                                'learning_rate': 0.001,
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'seed': 0,
                                'hidden_size': 1000,
                                'hidden_layers': 2,
                                'no_experiences': 10,
                                'task_incremental': False,
                                'log_path': './logs/s_mnist/wa/'}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.classic.SplitMNIST(n_experiences=args.no_experiences, return_task_id=args.task_incremental)

    #model = MultiHeadMLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers) if args.task_incremental \
     #   else MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers)
    model = SimpleMLP()

    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()
    csv_logger = avl.logging.CSVLogger(log_folder=args.log_path + 'csv_log*')
    text_logger = avl.logging.TextLogger(open(args.log_path + 'log.txt', 'w'))
    tensorboard_logger = avl.logging.TensorboardLogger(tb_log_dir=args.log_path + 'tensor_log*')

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        metrics.timing_metrics(epoch=True),
        metrics.forgetting_metrics(experience=True, stream=True),
        metrics.confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=False, stream=True),
        loggers=[interactive_logger, csv_logger, text_logger, tensorboard_logger])

    cl_strategy = avl.training.Naive(
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=args.eval_mb_size,
        device=device, evaluator=evaluation_plugin, plugins=[WeightAveragingPlugin()])

    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        cl_strategy.train(experience)
        result = cl_strategy.eval(benchmark.test_stream)

    return result


if __name__ == '__main__':
    res = wa_smnist()
    print(res)
