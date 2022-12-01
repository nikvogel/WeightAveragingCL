"""
This script shows how to run an experiment on a specific strategy and benchmark.
You can override default parameters by providing a dictionary as input to the method.
You can find all the parameters used by the experiment in the source file of the experiment.
"""

# select the experiment
from experiments.permuted_mnist import naive_pmnist, wa_pmnist, ewc_pmnist, cumulative_pmnist
from experiments.split_mnist import naive_smnist, wa_smnist, cumulative_smnist
from experiments.permuted_mnist_simple import naive_pmnist_simple, wa_pmnist_simple, \
                                                ewc_pmnist_simple, cumulative_pmnist_simple


naive_pmnist_simple()
wa_pmnist_simple()
ewc_pmnist_simple()
cumulative_pmnist_simple()


