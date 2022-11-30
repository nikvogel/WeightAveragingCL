import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MLP
from experiments.utils import set_seed, create_default_args


def ewc_pmnist(override_args=None):

    args = create_default_args({'cuda': 1,
                                'epochs': 10,
                                'learning_rate': 0.001,
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'seed': 0,
                                'dropout': 0,
                                'ewc_lambda': 1,
                                'ewc_mode': 'separate',
                                'ewc_decay': None,
                                'hidden_size': 1000,
                                'hidden_layers': 2,
                                'no_experiences': 10,
                                'log_path': './logs/p_mnist/ewc/'}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.PermutedMNIST(args.no_experiences)
    model = MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                drop_rate=args.dropout)
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

    cl_strategy = avl.training.EWC(
        model, SGD(model.parameters(), lr=args.learning_rate), criterion,
        ewc_lambda=args.ewc_lambda, mode=args.ewc_mode, decay_factor=args.ewc_decay,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=args.eval_mb_size,
        device=device, evaluator=evaluation_plugin)

    result = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        result = cl_strategy.eval(benchmark.test_stream)

    return result


if __name__ == '__main__':
    res = ewc_pmnist()
    print(res)