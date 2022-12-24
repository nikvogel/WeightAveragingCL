from experiments.permuted_mnist_update import wa_pmnist_update
import telegram_send
import time

epochs = [1, 2, 5, 10]
hidden_sizes = [100, 500, 1000, 2000]
hidden_layers = [1, 2, 5]
no_experiences = [5, 10, 20]
weighting_methods = ['average', 'inverted_t', 'inverted_t_adapted']

telegram_send.send(messages=[f"\U00002705 Started - {time.strftime('%d.%m.%Y %H:%M:%S', time.localtime())}"])

for epoch in epochs:
    overwrite_dict = {'epochs': epoch,
                      'log_path': f'./logs/wa_exp_pmnist_update/epochs/{epoch}/'}
    wa_pmnist_update(overwrite_dict)
telegram_send.send(meesages=["\U000023F3 Epochs done - {time.strftime('%d.%m.%Y %H:%M:%S', time.localtime())}."])

for hidden_size in hidden_sizes:
    overwrite_dict = {'hidden_size': hidden_size,
                      'log_path': f'./logs/wa_exp_pmnist_update/hidden_size/{hidden_size}/'}
    wa_pmnist_update(overwrite_dict)
telegram_send.send(meesages=["\U000023F3 Hidden size done - {time.strftime('%d.%m.%Y %H:%M:%S', time.localtime())}."])

for hidden_layer in hidden_layers:
    overwrite_dict = {'hidden_layers': hidden_layer,
                      'log_path': f'./logs/wa_exp_pmnist_update/hidden_layers/{hidden_layer}/'}
    wa_pmnist_update(overwrite_dict)
telegram_send.send(meesages=["\U000023F3 Hidden layer done - {time.strftime('%d.%m.%Y %H:%M:%S', time.localtime())}."])

for no_experience in no_experiences:
    overwrite_dict = {'no_experiences': no_experience,
                      'log_path': f'./logs/wa_exp_pmnist_update/experiences/{no_experience}/'}
    wa_pmnist_update(overwrite_dict)
telegram_send.send(meesages=["\U000023F3 No experiences done - {time.strftime('%d.%m.%Y %H:%M:%S', time.localtime())}."])

for weighting_method in weighting_methods:
    overwrite_dict = {'weighting_method': weighting_method,
                      'log_path': f'./logs/wa_exp_pmnist_update/weighting_method/{weighting_method}/'}
    wa_pmnist_update(overwrite_dict)
telegram_send.send(meesages=["\U000023F3 Weighting methods done - {time.strftime('%d.%m.%Y %H:%M:%S', time.localtime())}."])
telegram_send.send(messages=[f'\U00009989 Finished.'])
