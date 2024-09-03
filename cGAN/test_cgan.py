import os
import yaml
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, os.path.abspath(parent_dir))
from modules.data_manipulation import preprocess_data_for_model, inverse_scaling
from modules.data_loading import load_preprocessed_data
import modules.model_definitions_pytorch as md
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from modules.plotting import plot_gan_samples
from scipy.stats import wasserstein_distance
import warnings
import fatpack
from scipy.spatial.distance import pdist, squareform
import pywt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'params_test_cgan.yaml'), 'r') as file:
    params = dict(yaml.safe_load(file))

main_model_folder = params.get('model_dir')
modelnames = params.get('model')


# Empty lists for metrics
mse_loss_list_multiple_models = []
fft_mse_loss_list_multiple_models, fft_log_mse_loss_list_multiple_models = [], []
wasserstein_distance_loss_list_multiple_models, fft_wasserstein_distance_loss_list_multiple_models = [], []
generator_diversity_list_multiple_models = []
wt_cA_mse_loss_list_multiple_models, wt_cD1_mse_loss_list_multiple_models, wt_cD2_mse_loss_list_multiple_models, wt_cD3_mse_loss_list_multiple_models = [], [], [], []
fatigue_error_multiple_models = []

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

    # Empty lists for metrics
    mse_loss_list = []
    fft_mse_loss_list, fft_log_mse_loss_list = [], []
    wasserstein_distance_loss_list, fft_wasserstein_distance_loss_list = [], []
    generator_diversity_list = []
    wt_cA_mse_loss_list, wt_cD1_mse_loss_list, wt_cD2_mse_loss_list, wt_cD3_mse_loss_list = [], [], [], []
    fatigue_list_pred, fatigue_list_true = [], []


    for epoch in params.get('cp'):
        print(f"Epoch: {epoch}")
        weigths_path = os.path.join(main_model_folder, modelname, "cp", f"cp-{epoch:04d}.pth")
        if device==torch.device('cuda'):
            generator.load_state_dict(torch.load(weigths_path, weights_only=False))
        else:
            generator.load_state_dict(torch.load(weigths_path, weights_only=False, map_location=torch.device('cpu')))

        Y_sample_list, Y_fft_sample_list, Y_gen_sample_list, Y_gen_fft_sample_list = [], [], [], []
        sample_tested_size = 0
        fft_mse_loss = 0
        fft_log_mse_loss = 0
        wasserstein_distance_loss = 0
        fft_wasserstein_distance_loss = 0
        mse_loss = 0
        fatigue_list_epoch_pred, fatigue_list_epoch_true = [], []
        generator_diversity_epoch = 0
        wt_cA_mse_loss, wt_cD1_mse_loss, wt_cD2_mse_loss, wt_cD3_mse_loss = 0, 0, 0, 0

        for i, (X_i, Y_i) in enumerate(dataloader):
            X_i = X_i.to(device).float()
            current_batch_size = Y_i.size(0)
            sample_tested_size += current_batch_size

            Y_gen_i = generator(X_i)
            Y_gen_i = Y_gen_i.detach().cpu().numpy().squeeze()
            Y_i = Y_i.detach().cpu().numpy().squeeze()
            Y_i = inverse_scaling(Y_i, scaler=scaler_y[0])
            Y_gen_i = inverse_scaling(Y_gen_i, scaler=scaler_y[0])

            mse_loss += np.sum((Y_i - Y_gen_i)**2)

            # FFT
            if params.get('compute_fft'):
                Y_gen_i_fft = np.abs(np.fft.fft(Y_gen_i, axis=1))[:, :timesteps//2]
                Y_i_fft = np.abs(np.fft.fft(Y_i, axis=1))[:, :timesteps//2]
                # Remove the first frequency component, otherwise it will dominate the loss
                Y_gen_i_fft = Y_gen_i_fft[:, 1:]
                Y_i_fft = Y_i_fft[:, 1:]
                fft_mse_loss += np.sum((Y_i_fft - Y_gen_i_fft)**2)
                fft_log_mse_loss += np.sum((np.log(Y_i_fft) - np.log(Y_gen_i_fft))**2)

            if params.get('compute_wavelets'):
                wt_Y_i_cA, wt_Y_i_cD1, wt_Y_i_cD2, wt_Y_i_cD3 = pywt.wavedec(Y_i, params.get('wavelet_function'), level=3)
                wt_Y_gen_i_cA, wt_Y_gen_i_cD1, wt_Y_gen_i_cD2, wt_Y_gen_i_cD4 = pywt.wavedec(Y_gen_i, params.get('wavelet_function'), level=3)
                wt_cA_mse_loss += np.sum((wt_Y_i_cA - wt_Y_gen_i_cA)**2)
                wt_cD1_mse_loss += np.sum((wt_Y_i_cD1 - wt_Y_gen_i_cD1)**2)
                wt_cD2_mse_loss += np.sum((wt_Y_i_cD2 - wt_Y_gen_i_cD2)**2)
                wt_cD3_mse_loss += np.sum((wt_Y_i_cD3 - wt_Y_gen_i_cD4)**2)           

            if params.get('compute_wasserstein'):
                for k in range(current_batch_size):
                    wasserstein_distance_loss += wasserstein_distance(Y_i[k], Y_gen_i[k])

            # Fatigue
            if params.get('compute_fatigue'):
                for k in range(current_batch_size):
                    S_true = fatpack.find_rainflow_ranges(Y_i[k])
                    fatigue_true = curve.find_miner_sum(S_true)
                    fatigue_list_epoch_true.append(fatigue_true)

                    try:
                        S_pred = fatpack.find_rainflow_ranges(Y_gen_i[k])
                        fatigue_pred = curve.find_miner_sum(S_pred)
                    except IndexError as e:
                        print(f"IndexError encountered at index {k}: {e}. Returning 0.")
                        fatigue_pred = 0.0
                    fatigue_list_epoch_pred.append(fatigue_pred)
                
            # Statistics on timeseries
            if params.get('plot_samples'):
                if any([i in indices for i in range(sample_tested_size - current_batch_size, sample_tested_size)]):
                    indices_in_range = [i for i in range(sample_tested_size - current_batch_size, sample_tested_size) if i in indices]
                    for idx in indices_in_range:
                        idx = idx % current_batch_size
                        Y_sample_list.append(Y_i[idx])
                        Y_gen_sample_list.append(Y_gen_i[idx])
                        if params.get('compute_fft'):
                            Y_fft_sample_list.append(Y_i_fft[idx])
                            Y_gen_fft_sample_list.append(Y_gen_i_fft[idx])
                            if params.get('compute_wasserstein'):
                                fft_wasserstein_distance_loss += wasserstein_distance(Y_i_fft[idx], Y_gen_i_fft[idx])

            if params.get('compute_generator_diversity'):
                generated_series = np.array([generator(X_i).detach().cpu().numpy() for _ in range(10)])
                time_series_matrix = np.array([series.flatten() for series in generated_series])
                distance_matrix = squareform(pdist(time_series_matrix, metric='euclidean'))
                generator_diversity = np.mean(distance_matrix)
                generator_diversity_epoch += generator_diversity
                


        mse_loss_list.append(mse_loss/num_samples)            
        fft_mse_loss_list.append(fft_mse_loss/num_samples)
        fft_log_mse_loss_list.append(fft_log_mse_loss/num_samples)
        wasserstein_distance_loss_list.append(wasserstein_distance_loss/num_samples)
        fft_wasserstein_distance_loss_list.append(fft_wasserstein_distance_loss/num_samples)
        fatigue_list_pred.append(fatigue_list_epoch_pred)
        fatigue_list_true.append(fatigue_list_epoch_true)
        generator_diversity_list.append(generator_diversity_epoch/num_samples)
        wt_cA_mse_loss_list.append(wt_cA_mse_loss/num_samples)
        wt_cD1_mse_loss_list.append(wt_cD1_mse_loss/num_samples)
        wt_cD2_mse_loss_list.append(wt_cD2_mse_loss/num_samples)
        wt_cD3_mse_loss_list.append(wt_cD3_mse_loss/num_samples)

        if params.get('plot_samples'):
            plot_gan_samples(Y_sample_list, Y_gen_sample_list, x=t, num_pairs=8, figsize=(3, 5),
                            plot_name=f'ts_e{epoch}',
                            output_folder=os.path.join(output_folder, "testing"))
            if params.get('compute_fft'):
                plot_gan_samples(Y_fft_sample_list, Y_gen_fft_sample_list, x=fft_freq, num_pairs=8, figsize=(3, 5),
                                plot_name=f'fft_e{epoch}',
                                output_folder=os.path.join(output_folder, "testing"))   
                plot_gan_samples(Y_fft_sample_list, Y_gen_fft_sample_list, x=fft_freq, num_pairs=8, figsize=(3, 5),
                                plot_name=f'fft_log_e{epoch}',
                                output_folder=os.path.join(output_folder, "testing"), log=True)
                
    fatigue_arr_true = np.array(fatigue_list_true)
    fatigue_arr_pred = np.array(fatigue_list_pred)
    fatigue_error_per_epoch = np.mean(np.abs(fatigue_arr_true - fatigue_arr_pred), axis=1)

    def plot_value_per_epoch(epoch_list, value_list, value_name, plot_name, output_folder, figsize=(5, 4)):
        plt.figure(figsize=figsize)
        plt.plot(epoch_list, value_list)
        plt.xlabel("Epoch")
        plt.ylabel(value_name)
        plt.title(f"{value_name} per Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{plot_name}.pdf"))
        plt.close()

    plot_value_per_epoch(params.get('cp'), fatigue_error_per_epoch, "Fatigue Error", "fatigue_error", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), mse_loss_list, "MSE Loss", "ts_mse_loss", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), fft_mse_loss_list, "MSE Loss", "fft_mse_loss", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), fft_log_mse_loss_list, "MSE Loss", "fft_log_mse_loss", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), wasserstein_distance_loss_list, "Wasserstein Distance", "wasserstein_distance_loss", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), fft_wasserstein_distance_loss_list, "Wasserstein Distance", "fft_wasserstein_distance_loss", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), generator_diversity_list, "Generator Diversity", "generator_diversity", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), wt_cA_mse_loss_list, "Wavelet cA MSE Loss", "wt_cA_mse_loss", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), wt_cD1_mse_loss_list, "Wavelet cD1 MSE Loss", "wt_cD1_mse_loss", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), wt_cD2_mse_loss_list, "Wavelet cD2 MSE Loss", "wt_cD2_mse_loss", os.path.join(output_folder, "testing"))
    plot_value_per_epoch(params.get('cp'), wt_cD3_mse_loss_list, "Wavelet cD3 MSE Loss", "wt_cD3_mse_loss", os.path.join(output_folder, "testing"))

    mse_loss_list_multiple_models.append(mse_loss_list)
    fft_mse_loss_list_multiple_models.append(fft_mse_loss_list)
    fft_log_mse_loss_list_multiple_models.append(fft_log_mse_loss_list)
    wasserstein_distance_loss_list_multiple_models.append(wasserstein_distance_loss_list)
    fft_wasserstein_distance_loss_list_multiple_models.append(fft_wasserstein_distance_loss_list)
    generator_diversity_list_multiple_models.append(generator_diversity_list)
    wt_cA_mse_loss_list_multiple_models.append(wt_cA_mse_loss_list)
    wt_cD1_mse_loss_list_multiple_models.append(wt_cD1_mse_loss_list)
    wt_cD2_mse_loss_list_multiple_models.append(wt_cD2_mse_loss_list)
    wt_cD3_mse_loss_list_multiple_models.append(wt_cD3_mse_loss_list)
    fatigue_error_multiple_models.append(fatigue_error_per_epoch)


def plot_value_per_epoch_multiple_models(epoch_list, value_list, value_name, plot_name, 
                                         output_folder, figsize=(5, 4), modelnames=None,
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

# Plots comparing different models
if len(modelnames)>1:
    modelnames = params.get('model_labels')
    if not os.path.exists(os.path.join(main_model_folder, "testing")):
        os.makedirs(os.path.join(main_model_folder, "testing"))

    plot_value_per_epoch_multiple_models(params.get('cp'), mse_loss_list_multiple_models, "MSE Loss", "ts_mse_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), mse_loss_list_multiple_models, "MSE Loss (log)", "ts_mse_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)

    plot_value_per_epoch_multiple_models(params.get('cp'), fft_mse_loss_list_multiple_models, "FFT MSE Loss", "fft_mse_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), fft_mse_loss_list_multiple_models, "FFT MSE Loss (log)", "fft_mse_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)

    plot_value_per_epoch_multiple_models(params.get('cp'), fft_log_mse_loss_list_multiple_models, "FFT log MSE Loss", "fft_log_mse_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), fft_log_mse_loss_list_multiple_models, "FFT log MSE Loss (log)", "fft_log_mse_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    
    plot_value_per_epoch_multiple_models(params.get('cp'), wasserstein_distance_loss_list_multiple_models, "Wasserstein Distance", "wasserstein_distance_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), wasserstein_distance_loss_list_multiple_models, "Wasserstein Distance (log)", "wasserstein_distance_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    
    plot_value_per_epoch_multiple_models(params.get('cp'), fft_wasserstein_distance_loss_list_multiple_models, "Wasserstein Distance", "fft_wasserstein_distance_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), fft_wasserstein_distance_loss_list_multiple_models, "Wasserstein Distance (log)", "fft_wasserstein_distance_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    
    plot_value_per_epoch_multiple_models(params.get('cp'), generator_diversity_list_multiple_models, "Generator Diversity", "generator_diversity", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), generator_diversity_list_multiple_models, "Generator Diversity (log)", "generator_diversity_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    
    plot_value_per_epoch_multiple_models(params.get('cp'), wt_cA_mse_loss_list_multiple_models, "Wavelet cA MSE Loss", "wt_cA_mse_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), wt_cA_mse_loss_list_multiple_models, "Wavelet cA MSE Loss (log)", "wt_cA_mse_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    
    plot_value_per_epoch_multiple_models(params.get('cp'), wt_cD1_mse_loss_list_multiple_models, "Wavelet cD1 MSE Loss", "wt_cD1_mse_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), wt_cD1_mse_loss_list_multiple_models, "Wavelet cD1 MSE Loss (log)", "wt_cD1_mse_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    
    plot_value_per_epoch_multiple_models(params.get('cp'), wt_cD2_mse_loss_list_multiple_models, "Wavelet cD2 MSE Loss", "wt_cD2_mse_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), wt_cD2_mse_loss_list_multiple_models, "Wavelet cD2 MSE Loss (log)", "wt_cD2_mse_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    
    plot_value_per_epoch_multiple_models(params.get('cp'), wt_cD3_mse_loss_list_multiple_models, "Wavelet cD3 MSE Loss", "wt_cD3_mse_loss", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), wt_cD3_mse_loss_list_multiple_models, "Wavelet cD3 MSE Loss (log)", "wt_cD3_mse_loss_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    
    plot_value_per_epoch_multiple_models(params.get('cp'), fatigue_error_multiple_models, "Fatigue Error", "fatigue_error", os.path.join(main_model_folder, "testing"), modelnames=modelnames)
    plot_value_per_epoch_multiple_models(params.get('cp'), fatigue_error_multiple_models, "Fatigue Error (log)", "fatigue_error_log", os.path.join(main_model_folder, "testing"), modelnames=modelnames, ylog=True)
    