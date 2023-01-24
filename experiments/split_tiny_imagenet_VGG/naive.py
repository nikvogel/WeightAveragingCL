import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from avalanche.evaluation import metrics as metrics
from models import MultiHeadVGGSmall
from experiments.utils import set_seed, create_default_args


def naive_stinyimagenet(override_args=None):
    """
    "Learning without Forgetting" by Li et. al. (2016).
    http://arxiv.org/abs/1606.09282
    Since experimental setup of the paper is quite outdated and not
    easily reproducible, this experiment is based on
    "A continual learning survey: Defying forgetting in classification tasks"
    De Lange et. al. (2021).
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9349197

    We use a VGG network, which leads a lower performance than the one from
    De Lange et. al. (2021).
    """
    args = create_default_args({'cuda': 0,
                                'epochs': 20,
                                'hidden_size': 512,
                                'learning_rate': 0.001,
                                'optimizer': 'Adam',
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'no_experiences': 10,
                                'log_path': './logs/s_tiny_imagenet/naive/',
                                'seed': 0,
                                'dataset_root': None}, override_args)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitTinyImageNet(
        args.no_experiences, return_task_id=True, dataset_root=args.dataset_root)
    model = MultiHeadVGGSmall(n_classes=200, hidden_size=args.hidden_size)
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

    cl_strategy = avl.training.Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=args.eval_mb_size,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    res = naive_stinyimagenet()
    print(res)
