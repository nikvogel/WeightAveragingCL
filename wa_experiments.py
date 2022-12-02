from experiments.permuted_mnist import wa_pmnist

epochs = [1, 2, 5, 10]
learning_rates = [0.01, 0.005, 0.001]
hidden_sizes = [100, 500, 1000, 2000]
hidden_layers = [1, 2, 5, 10]
no_experiences = [5, 10, 20]
weighting_methods = ['average', 'inverted_t', 'inverted_t_adapted']


for epoch in epochs:
    overwrite_dict = {'epochs': epoch,
                      'log_path': f'./logs/wa_exp_pmnist/epochs/{epoch}/'}
    wa_pmnist(overwrite_dict)

for learning_rate in learning_rates:
    overwrite_dict = {'learning_rate': learning_rate,
                      'log_path': f'./logs/wa_exp_pmnist/learning_rate/{learning_rate}/'}
    wa_pmnist(overwrite_dict)

for hidden_size in hidden_sizes:
    overwrite_dict = {'hidden_size': hidden_size,
                      'log_path': f'./logs/wa_exp_pmnist/hidden_size/{hidden_size}/'}
    wa_pmnist(overwrite_dict)

for hidden_layer in hidden_layers:
    overwrite_dict = {'hidden_layers': hidden_layer,
                      'log_path': f'./logs/wa_exp_pmnist/hidden_layers/{hidden_layer}/'}
    wa_pmnist(overwrite_dict)

for no_experience in no_experiences:
    overwrite_dict = {'no_experiences': no_experience,
                      'log_path': f'./logs/wa_exp_pmnist/experiences/{no_experience}/'}
    wa_pmnist(overwrite_dict)

for weighting_method in weighting_methods:
    overwrite_dict = {'weighting_method': weighting_method,
                      'log_path': f'./logs/wa_exp_pmnist/weighting_method/{weighting_method}/'}
    wa_pmnist(overwrite_dict)
