import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from avalanche.evaluation import metrics as metrics
from avalanche.models import SimpleMLP, as_multitask, MTSimpleMLP, MultiHeadClassifier
from models import MLP, MultiHeadMLP, WeightAveragingPlugin
from experiments.utils import set_seed, create_default_args

def wa_smnist(override_args=None):

    args = create_default_args({'cuda': 0,
                                'epochs': 10,
                                'layers': 2, 
                                'hidden_size': 512,
                                'learning_rate': 0.001, 
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'no_experiences': 5,
                                'task_incremental': True,
                                'wa_alpha': 1, 
                                'log_path': './logs/s_mnist/wa/',
                                'seed': 0}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitMNIST(5, shuffle=False, return_task_id=args.task_incremental, class_ids_from_zero_in_each_exp=True)
    model = MultiHeadMLP(hidden_size=args.hidden_size, hidden_layers=args.layers)
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
        train_mb_size=args.train_mb_size, train_epochs=args.epochs,
        device=device, evaluator=evaluation_plugin, plugins=[WeightAveragingPlugin(args.wa_alpha)])

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = wa_smnist()
    print(res)
