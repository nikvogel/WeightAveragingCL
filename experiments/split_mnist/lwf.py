import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from avalanche.evaluation import metrics as metrics
from avalanche.models import SimpleMLP, as_multitask, MTSimpleMLP, MultiHeadClassifier
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


def lwf_smnist(override_args=None):
    
    args = create_default_args({'cuda': 0,
                                'lwf_alpha': 1,
                                'lwf_temperature': 2, 
                                'epochs': 10,
                                'layers': 2, 
                                'hidden_size': 512,
                                'learning_rate': 0.001, 
                                'train_mb_size': 256,
                                'eval_mb_size': 128,
                                'no_experiences': 5,
                                'task_incremental': True,
                                'log_path': './logs/s_mnist/lwf/',
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

    cl_strategy = LwFCEPenalty(
        model, Adam(model.parameters(), lr=args.learning_rate), criterion,
        alpha=args.lwf_alpha, temperature=args.lwf_temperature,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)
        new_dict = cl_strategy.model.state_dict()
        print('new dict')
        for key in new_dict.keys():
            print(key)
            print(new_dict[key].size())

    return res


if __name__ == '__main__':
    res = lwf_smnist()
    print(res)
