from experiments.permuted_mnist import wa_pmnist

epochs = [1, 2, 5, 10]
learning_rates = [0.01, 0.005, 0.001]
hidden_sizes = [100, 500, 1000, 2000]
hidden_layers = [1, 2, 5, 10]
no_experiences = [5, 10, 20]
weighting_methods = ['average', 'inverted_t', 'inverted_t_adapted']

for no_experience in no_experiences:
    for epoch in epochs:
        for learning_rate in learning_rates:
            for hidden_size in hidden_sizes:
                for hidden_layer in hidden_layers:
                    for weighting_method in weighting_methods:
                        overwrite_dict = {'epochs': epoch,
                                          'learning_rate': learning_rate,
                                          'hidden_size': hidden_size,
                                          'hidden_layers': hidden_layer,
                                          'no_experiences': 10,
                                          'weighting_method': weighting_method,
                                          'log_path': f'./logs/wa_exp_pmnist/ex{no_experience}_ep{epoch}_lr'\
                                                      f'{learning_rate}_hs{hidden_size}_hl{hidden_layer}'\
                                                      f'_{weighting_method}'}
                        wa_pmnist(overwrite_dict)
