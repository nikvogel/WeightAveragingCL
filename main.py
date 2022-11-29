"""
This script shows how to run an experiment on a specific strategy and benchmark.
You can override default parameters by providing a dictionary as input to the method.
You can find all the parameters used by the experiment in the source file of the experiment.
"""

# select the experiment
import time
from experiments.permuted_mnist import wa_pmnist


start = time.time()
print('Start')
#naive_pmnist()

wa_pmnist()

end = time.time()
print(f'Experiment took: {end - start}s')