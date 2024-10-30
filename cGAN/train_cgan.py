import os
import yaml
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from timeit import default_timer as timer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, os.path.abspath(parent_dir))
from modules.dir_functions import timestamp_now
from modules.log_functions import produce_training_log
from modules.data_manipulation import preprocess_data_for_model, subset_by_simulation_conditions
from modules.data_loading import load_preprocessed_data
from modules import log_functions
from modules.plotting import plot_gan_samples
from modules.train_utils import get_optimizer
import modules.model_definitions_pytorch as md
from modules.losses import compute_g_loss, compute_d_loss
import argparse

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Pass YAML configuration to the script.")
    parser.add_argument('-p', '--params', required=True, help='Path to the YAML parameters file')
    return parser.parse_args()

# Parameter utilities
def par(key, default=None):
    if key not in params and default is None:
        raise KeyError(f"Parameter '{key}' not found in params")
    return params.get(key, default)

def load_params(params_file):
    with open(os.path.join(current_dir, params_file), 'r') as file:
        return yaml.safe_load(file)

# Directory setup    
def setup_output_folder():
    output_folder = f"{par('save_model_dir')}/cgan_{timestamp_now()}"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "cp"), exist_ok=True)
    return output_folder

# Data loading and preprocessing
def load_and_preprocess_data():
    if isinstance(par('datasets'), list):
        X_list, Y_list, info_list = [], [], []
        for data_dir in par('datasets'):
            X, Y, scaler_X, scaler_y, columns_X, columns_Y, info, info_columns = load_preprocessed_data(data_dir, add_info_to_X=par('add_info_to_X'))
            X, Y, scaler_X, scaler_y, info = preprocess_data_for_model(X, Y, scaler_X, scaler_y, col_names_X=columns_X, col_names_Y=columns_Y, info=info, params=params)
            X_list.append(X); Y_list.append(Y); info_list.append(info)
        X, Y, info = np.vstack(X_list), np.vstack(Y_list), np.vstack(info_list)
    else:
        raise ValueError("The 'datasets' parameter must be a list of dataset directories.")

    X, Y, info = subset_by_simulation_conditions(X, Y, params, info, info_columns, in_sample=True)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y, info

# Model loading
def load_models(condition_dim, output_channels):

    generator = md.Generator(input_features=condition_dim, hidden_dim=par('hidden_dim'), 
                             noise_dim=par('noise_vector_dim'), timesteps=par('t_len'))
    discriminator = (
        md.DiscriminatorWithSkipConnections(output_channels, condition_dim)
        if par('skip_connections') and par('discriminator_input') == "ts"
        else md.DiscriminatorFFTInputWithSkipConnections(output_channels, condition_dim, use_log=par('discriminator_use_log'))
        if par('skip_connections') and par('discriminator_input') == "fft"
        else md.Discriminator(output_channels, condition_dim)
        if par('discriminator_input') == "ts"
        else md.DiscriminatorFFTInput(output_channels, condition_dim, use_log=par('discriminator_use_log'))
    )

    return generator.to(device), discriminator.to(device)

# Loading pretrained weights if specified
def load_weights(generator, discriminator):
    if (par('load_generator_path') is not None) and (par('load_generator_cp') is not None):
        generator.load_state_dict(torch.load(os.path.join(par('load_generator_path'), "cp", f"cp-{par('load_generator_cp'):04d}.pth"), map_location=device))

    if (par('load_discriminator_path') is not None) and (par('load_discriminator_cp') is not None):
        discriminator.load_state_dict(torch.load(os.path.join(par('load_discriminator_path'), "cp", f"cp_discr-{par('load_discriminator_cp'):04d}.pth"), map_location=device))

# Plotting loss curves
def plot_losses(d_loss_list, g_loss_list, output_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(d_loss_list), label='Discriminator loss')
    plt.plot(np.array(g_loss_list), label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_folder}/plots/losses_per_epoch.pdf')
    plt.close()  


# Main function to execute script
def main():

    args = parse_args()
    global params
    params = load_params(args.params)
    output_folder = setup_output_folder()

    logs_path = os.path.join(output_folder, "log.out")
    logs = open(logs_path, 'w')
    logs.write(log_functions.print_time("Program started"))

    with open(os.path.join(output_folder, 'params_model.yaml'), 'w') as file:
        yaml.dump(params, file, default_flow_style=False)

    X, Y, info = load_and_preprocess_data()
    generator, discriminator = load_models(condition_dim=X.shape[2], output_channels=Y.shape[2])
    load_weights(generator, discriminator)


    # Optimizers
    optimizer_G = get_optimizer(par("optimizer_G"), generator, par('lr_G'))
    optimizer_D = get_optimizer(par("optimizer_D"), discriminator, par('lr_D'))

    # Loss function
    adv_loss = nn.BCEWithLogitsLoss()

    # Create dataset and dataloader
    print(f"Training data shapes:\nX: {X.shape}\nY: {Y.shape}")
    dataloader = DataLoader(TensorDataset(X, Y), batch_size=par('batch_size'), shuffle=True, pin_memory=True, drop_last=True)

    # Training
    d_loss_list, g_loss_list = [], []
    epoch_start_training_D, epoch_start_training_G = 0, 0

    if (par('load_discriminator_path') is not None) and (par('load_discriminator_cp') is not None):
        epoch_start_training_D = par('discriminator_nr_freeze_epochs')
    if (par('load_generator_path') is not None) and (par('load_generator_cp') is not None):
        epoch_start_training_G = par('generator_nr_freeze_epochs')


    labels_real = torch.ones(par('batch_size'), 1, requires_grad=False).to(device)
    labels_fake = torch.zeros(par('batch_size'), 1, requires_grad=False).to(device)

    # Define learnable parameters for the weights (initialize with 1.0 for equal contribution initially)
    lambda_adv = par('lambda_adv')
    lambda_mse = par('lambda_mse')

    if par('sa_weights'):
        lambda_adv = torch.nn.Parameter(torch.tensor(0.0, device=device))  # log(s_adv) This is actually log_lambda_avd
        lambda_mse = torch.nn.Parameter(torch.tensor(0.0, device=device))  # log(s_mse) This is actually log_lambda_mse
        optimizer_w = torch.optim.Adam([lambda_adv, lambda_mse], lr=1e-3)


    d_loss = torch.tensor(0, device=device, dtype=torch.float32)

    for epoch in range(1, par('n_epochs')+1):
        t1 = timer()
        d_accum = torch.tensor(0, device=device, dtype=torch.float32)
        g_accum = torch.tensor(0, device=device, dtype=torch.float32)

        for i, (conditions, real_data) in enumerate(dataloader):
            real_data = real_data.to(device, non_blocking=True).float()
            conditions = conditions.to(device, non_blocking=True).float()

            optimizer_D.zero_grad()
            generated_data = generator(conditions)
            if par('discriminator_match_features'):
                labels_discriminator_fake, features_output_fake = discriminator(generated_data.detach(), conditions, return_features=par('discriminator_match_features'))
                labels_discriminator_real, features_output_real = discriminator(real_data, conditions, return_features=par('discriminator_match_features'))
            else:
                labels_discriminator_fake = discriminator(generated_data.detach(), conditions)
                labels_discriminator_real = discriminator(real_data, conditions)

            if lambda_adv > 0:
                d_loss = compute_d_loss(discriminator, real_data, generated_data, conditions,  labels_real, labels_fake, labels_discriminator_fake, labels_discriminator_real, 
                                        lambda_gp=par('lambda_gp'), adv_loss=adv_loss, use_wgan=par('use_wgan'), device=device)
                
                if epoch >= epoch_start_training_D:
                    d_loss.backward()
                    optimizer_D.step()            
            
            optimizer_G.zero_grad()
            if par('sa_weights'):
                optimizer_w.zero_grad()

            labels_discriminator_fake = discriminator(generated_data, conditions)
            g_loss, g_adv, g_mse, g_fft_mse, g_fft_wass = compute_g_loss(generated_data, real_data, 
                                                                        labels_discriminator_fake, labels_real, 
                                                                        lambda_adv, lambda_mse, 
                                                                        par('lambda_fftwass'), par('lambda_fftmse'),
                                                                        use_wgan=par('use_wgan'), device=device,
                                                                        adv_loss=adv_loss, lambda_adv=lambda_adv, lambda_mse=lambda_mse,
                                                                        sa_weights=par('sa_weights'))
            

            if epoch >= epoch_start_training_G:
                g_loss.backward()
                optimizer_G.step()
                if par('sa_weights'):
                    optimizer_w.step()

            d_accum += d_loss
            g_accum += g_loss
            
        t2 = timer()

        d_loss_list.append(d_accum.item())
        g_loss_list.append(g_accum.item())

        if epoch % par('log_freq')==0:
            log_out_string = produce_training_log(epoch, t1, t2, d_loss, g_loss, g_adv, g_mse, g_fft_mse, g_fft_wass, params)
            logs.write(log_out_string + '\n')
            print(log_out_string)

        if epoch % par('plot_freq')==0:
            plot_gan_samples(real_data[:8].detach().cpu().numpy(), generated_data[:8].detach().cpu().numpy(), 
                            num_pairs=8, figsize=(5, 5), plot_name=f'comparison_output_{epoch}',
                            output_folder=os.path.join(output_folder, "plots"))
                
        if epoch % par('cp_freq')==0:
            torch.save(generator.state_dict(), os.path.join(output_folder, "cp", f"cp-{epoch:04d}.pth"))
            if par('save_discriminator'):
                torch.save(discriminator.state_dict(), os.path.join(output_folder, "cp", f"cp_discr-{epoch:04d}.pth"))
            print(f"Checkpoint saved at epoch {epoch}")


    plot_losses(d_loss_list, g_loss_list, output_folder)

    logs.write(log_functions.print_time("Program ended"))
    logs.close()

# Entry point for the script
if __name__ == "__main__":
    main()