import json

import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from avalanche.evaluation import metrics as metrics
from avalanche.models import SimpleMLP, SimpleCNN
from experiments.utils import set_seed, create_default_args
from models import WeightAveragingPlugin, CNN, CNN_Single


def wa_pmnist_simple_CNN(override_args=None):

    args = create_default_args({'cuda': 1,
                                'epochs': 2,
                                'learning_rate': 0.01,
                                'optimizer': 'Adam',
                                'train_mb_size': 128,
                                'eval_mb_size': 128,
                                'seed': 0,
                                'no_experiences': 10,
                                'wa_alpha': 1,
                                'log_path': './logs/p_mnist_simple/wa_CNNx2/'}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and args.cuda >= 0
                          else "cpu")
    benchmark = avl.benchmarks.PermutedMNIST(args.no_experiences)
    model = CNN_Single(N = 2, num_classes = args.no_experiences)

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    
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
        model, optimizer, criterion, train_mb_size=args.train_mb_size, train_epochs=args.epochs,
        device=device, evaluator=evaluation_plugin, plugins=[WeightAveragingPlugin(args.wa_alpha)])

    result = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        result = cl_strategy.eval(benchmark.test_stream)

    #with open(args.log_path + 'result.json', 'w') as fp:
     #   json.dump(result, fp)

    return result


if __name__ == '__main__':
    res = wa_pmnist_simple_CNN()
    print(res)
