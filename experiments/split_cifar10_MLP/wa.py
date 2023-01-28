import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from avalanche.evaluation import metrics as metrics
from avalanche.models import SimpleMLP, as_multitask, MTSimpleMLP
from avalanche.benchmarks import SplitCIFAR10
from models import MLP, MultiHeadMLP, WeightAveragingPlugin, CNN, CNN_6
from experiments.utils import set_seed, create_default_args
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin

def wa_s_cifar(override_args=None):

    args = create_default_args({'cuda': 0,
                                'epochs': 10,
                                'layers': 1,
                                'hidden_size': 512,
                                'learning_rate': 0.001,
                                'optimizer': 'Adam', 
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'no_experiences': 5,
                                'task_incremental': False,
                                'wa_alpha': 1,
                                'log_path': './logs/split_cifar10_mlp/wa/',
                                'seed': 0}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    model = MLP(hidden_size = args.hidden_size, hidden_layers= args.layers, input_size=32 * 32 * 3)
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

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    cl_strategy = avl.training.Naive(
        model, optimizer, criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs,
        device=device, evaluator=evaluation_plugin, plugins=[WeightAveragingPlugin(args.wa_alpha)])

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = wa_s_cifar()
    print(res)
