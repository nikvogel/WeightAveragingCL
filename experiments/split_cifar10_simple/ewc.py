import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from avalanche.evaluation import metrics as metrics
from avalanche.models import SimpleMLP, as_multitask
from avalanche.benchmarks import SplitCIFAR10
from models import MLP, MultiHeadMLP, WeightAveragingPlugin
from experiments.utils import set_seed, create_default_args

def ewc_s_cifar(override_args=None):

    args = create_default_args({'cuda': 0,
                                'epochs': 10,
                                'layers': 1, 
                                'ewc_lambda': 1,
                                'hidden_size': 1000,
                                'learning_rate': 0.001, 
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'no_experiences': 5,
                                'task_incremental': False,
                                'log_path': './logs/split_cifar10/ewc/',
                                'seed': 0}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    model = SimpleMLP(input_size=32 * 32 * 3, num_classes=10)
    model = as_multitask(model, "classifier")
    benchmark = SplitCIFAR10(n_experiences=5, return_task_id=True)    
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
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs,
        device=device, evaluator=evaluation_plugin, ewc_lambda=args.ewc_lambda)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = ewc_s_cifar()
    print(res)
