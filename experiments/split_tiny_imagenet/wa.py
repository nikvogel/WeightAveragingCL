import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
from models import MultiHeadVGGSmall, WeightAveragingPlugin
from experiments.utils import set_seed, create_default_args


def wa_stinyimagenet(override_args=None):
    args = create_default_args({'cuda': 0, 
                                'epochs': 30,
                                'learning_rate': 1e-3, 
                                'train_mb_size': 200, 
                                'seed': 0,
                                'log_path': './logs/s_tiny_imagenet/wa/',
                                'dataset_root': None}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitTinyImageNet(
        10, return_task_id=True, dataset_root=args.dataset_root)
    model = MultiHeadVGGSmall(n_classes=200)
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
        model,
        Adam(model.parameters(), lr=args.learning_rate),
        criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin, plugins=[WeightAveragingPlugin()])

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    res = wa_stinyimagenet()
    print(res)
