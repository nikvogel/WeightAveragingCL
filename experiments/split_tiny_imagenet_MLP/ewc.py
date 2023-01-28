import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from avalanche.evaluation import metrics as metrics
from models import MultiHeadVGGSmall, MLP
from avalanche.models import as_multitask

from experiments.utils import set_seed, create_default_args


def ewc_stinyimagenet(override_args=None):

    args = create_default_args({'cuda': 0,
                                'epochs': 20,
                                'ewc_lambda': 1,
                                'hidden_size': 512,
                                'learning_rate': 0.001,
                                'layers': 2,
                                'optimizer': 'Adam',
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'no_experiences': 10,
                                'log_path': './logs/s_tiny_imagenet_mlp/ewc/',
                                'seed': 0,
                                'dataset_root': None}, override_args)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitTinyImageNet(
        args.no_experiences, return_task_id=True, dataset_root=args.dataset_root)
    model = MLP(input_size=64*64*3, output_size=200, hidden_size=args.hidden_size, hidden_layers=args.layers)
    model = as_multitask(model, "classifier")
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

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    cl_strategy = avl.training.EWC(
        model,
        optimizer,
        criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=args.eval_mb_size,
        device=device, evaluator=evaluation_plugin, ewc_lambda=args.ewc_lambda)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    res = ewc_stinyimagenet()
    print(res)
