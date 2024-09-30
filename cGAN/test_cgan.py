import os
import yaml
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, os.path.abspath(parent_dir))
from modules.data_manipulation import preprocess_data_for_model, inverse_scaling
from modules.data_loading import load_preprocessed_data
from modules.cgan_metrics import compute_generator_diversity, compute_fft_loss
import modules.model_definitions_pytorch as md
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from modules.plotting import plot_gan_samples
from scipy.stats import wasserstein_distance
import warnings
import fatpack
import pywt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'params_test_cgan.yaml'), 'r') as file:
    params = dict(yaml.safe_load(file))

main_model_folder = params.get('model_dir')
modelnames = params.get('model')


def plot_value_per_epoch(epoch_list, value_list, value_name, plot_name, output_folder, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    plt.plot(epoch_list, value_list)
    plt.xlabel("Epoch")
    plt.ylabel(value_name)
    plt.title(f"{value_name} per Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{plot_name}.pdf"))
    plt.close()


def plot_value_per_epoch_multiple_models(epoch_list, value_list, value_name, plot_name, output_folder, figsize=(5, 4), modelnames=None,
                                         ylog=False):
    plt.figure(figsize=figsize)
    for i, value_list_model in enumerate(value_list):
        plt.plot(epoch_list, value_list_model, label=modelnames[i])
    plt.xlabel("Epoch")
    plt.ylabel(value_name)
    if ylog:
        plt.yscale('log')
    plt.title(f"{value_name} per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{plot_name}.pdf"))
    plt.close()



# Empty lists for metrics
metrics_all_models = {
    'mse_loss': [],
    'fft_mse_loss': [],
    'fft_log_mse_loss': [],
    'wasserstein_distance_loss': [],
    'fft_wasserstein_distance_loss': [],
    'generator_diversity': [],
    'wt_cA_mse_loss': [],
    'wt_cD1_mse_loss': [],
    'wt_cD2_mse_loss': [],
    'wt_cD3_mse_loss': [],
    'fatigue_mse_loss': []
}

for modelname in modelnames:
    print(f"Testing model: {modelname}")
    output_folder = f"{main_model_folder}/{modelname}"

    with open(f'{output_folder}/params_model.yaml', 'r') as file:
        params_model = dict(yaml.safe_load(file))

    # Load data
    if isinstance(params.get('datasets'), list):
        X_list, Y_list = [], []
        for data_dir in params.get('datasets'):
            X, Y, scaler_X, scaler_y, columns_X, columns_Y, info, info_columns = load_preprocessed_data(data_dir, add_info_to_X=params_model.get('add_info_to_X'))
            X, Y, scaler_X, scaler_y, info = preprocess_data_for_model(X, Y, scaler_X, scaler_y, col_names_X=columns_X, col_names_Y=columns_Y, info=info, params=params_model)


            X_list.append(X)
            Y_list.append(Y)

        X = np.vstack(X_list)
        Y = np.vstack(Y_list)
    else:
        raise ValueError("The 'datasets' parameter must be a list of dataset directories.")

    variance_y = np.var(Y)
    checkpoint_dir = os.path.join(output_folder, "cp")

    num_samples = X.shape[0]
    timesteps = X.shape[1]
    condition_dim = X.shape[2]
    input_channels = Y.shape[2]

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    if device==torch.device('cuda'):
        X = X.to(device)
        Y = Y.to(device)

    print("Testing data shapes:")
    print(f"X: {X.shape}")
    print(f"Y: {Y.shape}")

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=params.get('batch_size'), shuffle=False, drop_last=False)
    nr_batches = len(dataloader)

    if not os.path.exists(os.path.join(output_folder, "testing")):
        os.makedirs(os.path.join(output_folder, "testing"))

    generator = md.Generator(input_features=X.shape[2], hidden_dim=100, noise_dim=params_model.get('noise_vector_dim'), 
                            timesteps=params_model.get('t_len'))
    generator.to(device)

    np.random.seed(42)
    indices = np.random.choice(len(Y), 8, replace=False)

    # 6 seconds of data
    t = np.linspace(0, 6, params_model.get('t_len'))   
    # Sampling interval is in seconds so the frequency is in Hz
    fft_freq = np.fft.fftfreq(params_model.get('t_len'), d=t[1]-t[0])[1:timesteps//2]


    if params.get('compute_fatigue'):
        Sc = params.get('fatigue_sc')
        curve = fatpack.TriLinearEnduranceCurve(Sc)

    metrics = {
        'mse_loss': [],
        'fft_mse_loss': [],
        'fft_log_mse_loss': [],
        'wasserstein_distance_loss': [],
        'fft_wasserstein_distance_loss': [],
        'generator_diversity': [],
        'wt_cA_mse_loss': [],
        'wt_cD1_mse_loss': [],
        'wt_cD2_mse_loss': [],
        'wt_cD3_mse_loss': [],
        'fatigue_mse_loss': []    
        }


    for epoch in params.get('cp'):
        print(f"Epoch: {epoch}")
        weigths_path = os.path.join(main_model_folder, modelname, "cp", f"cp-{epoch:04d}.pth")
        if device==torch.device('cuda'):
            generator.load_state_dict(torch.load(weigths_path, weights_only=False))
        else:
            generator.load_state_dict(torch.load(weigths_path, weights_only=False, map_location=torch.device('cpu')))



        Y_sample_list, Y_fft_sample_list, Y_gen_sample_list, Y_gen_fft_sample_list = ([] for _ in range(4))
        error_count_not_enough_cycles = 0
        sample_tested_size = 0

        metrics_accum = {
            'fft_mse_loss': 0,
            'fft_log_mse_loss': 0,
            'wasserstein_distance_loss': 0,
            'fft_wasserstein_distance_loss': 0,
            'mse_loss': 0,
            'generator_diversity': 0,
            'wt_cA_mse_loss': 0,
            'wt_cD1_mse_loss': 0,
            'wt_cD2_mse_loss': 0,
            'wt_cD3_mse_loss': 0,
            'fatigue_mse_loss': 0
        }


        for i, (X_i, Y_i) in enumerate(dataloader):
            X_i = X_i.to(device).float()
            current_batch_size = Y_i.size(0)
            sample_tested_size += current_batch_size

            Y_gen_i, Y_i = generator(X_i).detach().cpu().numpy().squeeze(), Y_i.detach().cpu().numpy().squeeze()
            Y_i, Y_gen_i = inverse_scaling(Y_i, scaler=scaler_y[0]), inverse_scaling(Y_gen_i, scaler=scaler_y[0])

            metrics_accum['mse_loss'] += np.sum((Y_i - Y_gen_i)**2)

            # FFT
            if params.get('compute_fft'):
                Y_i_fft, Y_gen_i_fft, fft_mse_loss, fft_log_mse_loss = compute_fft_loss(Y_i, Y_gen_i)
                metrics_accum['fft_mse_loss'] += fft_mse_loss
                metrics_accum['fft_log_mse_loss'] += fft_log_mse_loss

            if params.get('compute_wavelets'):
                # Perform wavelet decomposition for both original and generated data
                wt_Y_i, wt_Y_gen_i = pywt.wavedec(Y_i, params['wavelet_function'], level=3), pywt.wavedec(Y_gen_i, params['wavelet_function'], level=3)

                # Iterate over the decomposition levels (cA, cD1, cD2, cD3) and accumulate the losses
                for level, key in enumerate(['wt_cA_mse_loss', 'wt_cD1_mse_loss', 'wt_cD2_mse_loss', 'wt_cD3_mse_loss']):
                    metrics_accum[key] += np.sum((wt_Y_i[level] - wt_Y_gen_i[level])**2)        

            if params.get('compute_wasserstein'):
                for k in range(current_batch_size):
                    metrics_accum['wasserstein_distance_loss'] += wasserstein_distance(Y_i[k], Y_gen_i[k])

            # Fatigue
            if params.get('compute_fatigue'):
                for k in range(current_batch_size):
                    S_true = fatpack.find_rainflow_ranges(Y_i[k])
                    fatigue_true = curve.find_miner_sum(S_true)

                    try:
                        S_pred = fatpack.find_rainflow_ranges(Y_gen_i[k])
                        fatigue_pred = curve.find_miner_sum(S_pred)
                    except IndexError as e:
                        error_count_not_enough_cycles += 1
                        fatigue_pred = 0.0
                    metrics_accum['fatigue_mse_loss'] += np.sum((fatigue_true - fatigue_pred)**2)
                
            # Statistics on timeseries
            if params.get('plot_samples') and any([i in indices for i in range(sample_tested_size - current_batch_size, sample_tested_size)]):
                indices_in_range = [i for i in range(sample_tested_size - current_batch_size, sample_tested_size) if i in indices]
                for idx in indices_in_range:
                    idx = idx % current_batch_size
                    Y_sample_list.append(Y_i[idx])
                    Y_gen_sample_list.append(Y_gen_i[idx])
                    if params.get('compute_fft'):
                        Y_fft_sample_list.append(Y_i_fft[idx])
                        Y_gen_fft_sample_list.append(Y_gen_i_fft[idx])
                    if params.get('compute_fft') and params.get('compute_wasserstein'):
                        metrics_accum['fft_wasserstein_distance_loss'] += wasserstein_distance(Y_i_fft[idx], Y_gen_i_fft[idx])

            if params.get('compute_generator_diversity'):
                metrics_accum['generator_diversity'] += compute_generator_diversity(generator, X_i, num_samples=10, metric='euclidean')
                

        # Update the metrics dictionary
        for key in metrics_accum:
            metrics[key].append(metrics_accum[key] / num_samples)


        if params.get('plot_samples'):
            plot_gan_samples(Y_sample_list, Y_gen_sample_list, x=t, num_pairs=8, figsize=(3, 5),
                            plot_name=f'ts_e{epoch}', output_folder=os.path.join(output_folder, "testing"))
            
        if params.get('plot_samples') and params.get('compute_fft'):
            plot_gan_samples(Y_fft_sample_list, Y_gen_fft_sample_list, x=fft_freq, num_pairs=8, figsize=(3, 5),
                            plot_name=f'fft_e{epoch}', output_folder=os.path.join(output_folder, "testing"))   
            plot_gan_samples(Y_fft_sample_list, Y_gen_fft_sample_list, x=fft_freq, num_pairs=8, figsize=(3, 5),
                            plot_name=f'fft_log_e{epoch}', output_folder=os.path.join(output_folder, "testing"), log=True)
                
    if error_count_not_enough_cycles > 0:
        print(f"Error: Not enough cycles to compute fatigue for {error_count_not_enough_cycles}/{len(Y)} samples.")

    for metric_key, data in metrics.items():
        title = metric_key.replace('_', ' ').title()
        filename = metric_key.lower()
        plot_value_per_epoch(params.get('cp'), data, title, filename, os.path.join(output_folder, "testing"))

    for key in metrics:
        metrics_all_models[key].append(metrics[key])


# Plots comparing different models
if len(modelnames)>1:
    modelnames = params.get('model_labels')
    if not os.path.exists(os.path.join(main_model_folder, "testing")):
        os.makedirs(os.path.join(main_model_folder, "testing"))

    for metric_key in metrics_all_models:
        title = metric_key.replace('_', ' ').title()  # Convert key to title format

        plot_value_per_epoch_multiple_models(params.get('cp'), metrics_all_models[metric_key], title, f"{metric_key}_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
        plot_value_per_epoch_multiple_models(params.get('cp'), metrics_all_models[metric_key], f"{title} (log)", f"{metric_key}_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)