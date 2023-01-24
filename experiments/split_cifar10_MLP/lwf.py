import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from avalanche.evaluation import metrics as metrics
from avalanche.models import MLP, SimpleMLP, as_multitask, MTSimpleMLP
from avalanche.benchmarks import SplitCIFAR10
from models import MLP, MultiHeadMLP
from experiments.utils import set_seed, create_default_args


class LwFCEPenalty(avl.training.LwF):
    """This wrapper around LwF computes the total loss
    by diminishing the cross-entropy contribution over time,
    as per the paper
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf
    The loss is L_tot = (1/n_exp_so_far) * L_cross_entropy +
                        alpha[current_exp] * L_distillation
    """
    def _before_backward(self, **kwargs):
        self.loss *= float(1/(self.clock.train_exp_counter+1))
        super()._before_backward(**kwargs)


def lwf_s_cifar_complex(override_args=None):
    """
    "Learning without Forgetting" by Li et. al. (2016).
    http://arxiv.org/abs/1606.09282
    Since experimental setup of the paper is quite outdated and not
    easily reproducible, this experiment is based on
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf

    The hyper-parameter alpha controlling the regularization is increased over time, resulting
    in a regularization of  (1- 1/n_exp_so_far) * L_distillation
    """
    args = create_default_args({'cuda': 0,
                                'lwf_alpha': 1,
                                'lwf_temperature': 2, 
                                'epochs': 10,
                                'layers': 2, 
                                'hidden_size': 1024,
                                'learning_rate': 0.001,
                                'optimizer': 'Adam',
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'no_experiences': 5,
                                'task_incremental': True,
                                'log_path': './logs/split_cifar10_complex/lwf/',
                                'seed': 0}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    model = MLP(hidden_size = args.hidden_size, hidden_layers= args.layers, input_size=32 * 32 * 3)
    model = as_multitask(model, "classifier")
    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=args.task_incremental,
                                          fixed_class_order=list(range(10)))  
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

        
    cl_strategy = LwFCEPenalty(
        model, optimizer, criterion,
        alpha=args.lwf_alpha, temperature=args.lwf_temperature,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = lwf_s_cifar_complex()
    print(res)
