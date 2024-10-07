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
from modules.train_utils import get_optimizer, compute_gradient_penalty
import modules.model_definitions_pytorch as md
import modules.losses as losses
import argparse

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Pass YAML configuration to the script.")
parser.add_argument('-p', '--params', required=True, help='Path to the YAML parameters file')
args = parser.parse_args()
with open(os.path.join(script_dir, args.params), 'r') as file:
    params = dict(yaml.safe_load(file))
par = params.get

output_folder = f"{par('save_model_dir')}/cgan_{timestamp_now()}"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

logs_path = os.path.join(output_folder, "log.out")
logs = open(logs_path, 'w')
logs.write(log_functions.print_time("Program started"))

with open(os.path.join(output_folder, 'params_model.yaml'), 'w') as file:
    yaml.dump(params, file, default_flow_style=False)

# Load data
if isinstance(par('datasets'), list):
    X_list, Y_list, info_list = [], [], []
    for data_dir in par('datasets'):
        X, Y, scaler_X, scaler_y, columns_X, columns_Y, info, info_columns = load_preprocessed_data(data_dir, add_info_to_X=params.get('add_info_to_X'))
        X, Y, scaler_X, scaler_y, info = preprocess_data_for_model(X, Y, scaler_X, scaler_y, col_names_X=columns_X, col_names_Y=columns_Y, info=info, params=params)
        X_list.append(X)
        Y_list.append(Y)
        info_list.append(info)
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    info = np.vstack(info_list)
else:
    raise ValueError("The 'datasets' parameter must be a list of dataset directories.")

X, Y, info = subset_by_simulation_conditions(X, Y, params, info, info_columns, in_sample=True)

variance_y = np.var(Y)
cp_dir = os.path.join(output_folder, "cp")

num_samples = X.shape[0]
timesteps = X.shape[1]
condition_dim = X.shape[2]
input_channels = Y.shape[2] 

# Generator
generator = md.Generator(input_features=X.shape[2], hidden_dim=par('hidden_dim'), noise_dim=par('noise_vector_dim'), 
                         timesteps=par('t_len'))

# Load weights of pretrained generator
if (par('load_generator_path') is not None) and (par('load_generator_cp') is not None):
    weigths_path = os.path.join(par('load_generator_path'), "cp", f"cp-{par('load_generator_cp'):04d}.pth")
    generator.load_state_dict(torch.load(weigths_path, map_location=device))

# Discriminator
# TODO: This can load one model with different parameters.
if par('skip_connections'):
    if par('discriminator_input') == "ts":
        discriminator = md.DiscriminatorWithSkipConnections(input_channels, condition_dim)
    elif par('discriminator_input') == "fft":
        discriminator = md.DiscriminatorFFTInputWithSkipConnections(input_channels, condition_dim, use_log=par('discriminator_use_log'))
else:
    if par('discriminator_input') == "ts":
        discriminator = md.Discriminator(input_channels, condition_dim)
    elif par('discriminator_input') == "fft":
        discriminator = md.DiscriminatorFFTInput(input_channels, condition_dim, use_log=par('discriminator_use_log'))

# Load weights of pretrained discriminator
if (par('load_discriminator_path') is not None) and (par('load_discriminator_cp') is not None):
    weigths_path = os.path.join(par('load_discriminator_path'), "cp", f"cp_discr-{par('load_discriminator_cp'):04d}.pth")
    discriminator.load_state_dict(torch.load(weigths_path, map_location=device))

# Optimizers
optimizer_G = get_optimizer(par("optimizer_G"), generator, par('lr_G'))
optimizer_D = get_optimizer(par("optimizer_D"), discriminator, par('lr_D'))

# Loss function
adv_loss = nn.BCEWithLogitsLoss()

generator.to(device)
discriminator.to(device)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Create dataset and dataloader
print("Training data shapes:")
print(f"X: {X.shape}")
print(f"Y: {Y.shape}")
dataloader = DataLoader(TensorDataset(X, Y), batch_size=par('batch_size'), shuffle=True, pin_memory=True, drop_last=True)
nr_batches = len(dataloader)

if not os.path.exists(os.path.join(output_folder, "plots")):
    os.makedirs(os.path.join(output_folder, "plots"))
if not os.path.exists(cp_dir):
    os.makedirs(cp_dir)

# Training
d_loss_list, g_loss_list = [], []
fft_wass_loss, fft_mse = None, None
epoch_start_training_D, epoch_start_training_G = 0, 0

if (par('load_discriminator_path') is not None) and (par('load_discriminator_cp') is not None):
    epoch_start_training_D = par('discriminator_nr_freeze_epochs')
if (par('load_generator_path') is not None) and (par('load_generator_cp') is not None):
    epoch_start_training_G = par('generator_nr_freeze_epochs')

lambda_fftwass = par('lambda_fftwass', 0)
lambda_fftmse = par('lambda_fftmse', 0)
use_wgan = par('use_wgan', False)
lambda_gp = par('lambda_gp', 0)

labels_valid = torch.ones(par('batch_size'), 1, requires_grad=False).to(device)
labels_fake = torch.zeros(par('batch_size'), 1, requires_grad=False).to(device)

# Define learnable parameters for the weights (initialize with 1.0 for equal contribution initially)
lambda_adv = par('lambda_adv', 0)
lambda_mse = par('lambda_mse', 0)
normalized_lambda_adv = lambda_adv
normalized_lambda_mse = lambda_mse
if par('sa_weights'):
    lambda_adv = torch.nn.Parameter(torch.tensor(0.5, device=device))
    lambda_mse = torch.nn.Parameter(torch.tensor(0.5, device=device))

# Create an optimizer for the weights
optimizer_w = torch.optim.Adam([lambda_adv, lambda_mse], lr=1e-3)


def _compute_g_loss(generated_data, real_data, labels_discriminator_fake, labels_valid, 
                    normalized_lambda_adv, normalized_lambda_mse, 
                    lambda_fftwass, lambda_fftmse,
                    use_wgan=use_wgan, device=device):
    
    g_loss = torch.tensor(0, device=device, dtype=torch.float32)
    g_adv, g_mse, g_fft_mse, g_fft_wass = None, None, None, None

    if use_wgan:
        g_adv = normalized_lambda_adv * (-labels_discriminator_fake.mean())
        g_loss += g_adv

    elif normalized_lambda_adv != 0:
        g_adv = normalized_lambda_adv * adv_loss(labels_discriminator_fake, labels_valid)
        g_loss += g_adv

    if normalized_lambda_mse != 0:
        g_mse = normalized_lambda_mse * nn.MSELoss()(generated_data, real_data)
        g_loss += g_mse

    elif lambda_fftmse != 0:
        g_fft_mse = losses.FFTMSELoss()(generated_data, real_data)
        g_fft_mse = lambda_fftmse * g_fft_mse
        g_loss += g_fft_mse

    elif lambda_fftwass != 0:
        g_fft_wass = losses.FFTWassersteinLoss()(generated_data, real_data)
        g_fft_wass = lambda_fftwass * g_fft_wass
        g_loss += g_fft_wass

    return g_loss, g_adv, g_mse, g_fft_mse, g_fft_wass

def _compute_d_loss(labels_discriminator_fake, labels_discriminator_real, use_wgan=use_wgan, device=device):
    
    if use_wgan:
        d_loss = labels_discriminator_fake.mean() - labels_discriminator_real.mean()
        gradient_penalty = compute_gradient_penalty(discriminator, real_data, generated_data.detach(), conditions, device)
        d_loss += lambda_gp * gradient_penalty

    else:
        d_real_loss = adv_loss(labels_discriminator_real, labels_valid)
        d_fake_loss = adv_loss(labels_discriminator_fake, labels_fake)
        d_loss = (d_real_loss + d_fake_loss) / 2

    return d_loss

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
        labels_discriminator_fake = discriminator(generated_data.detach(), conditions)
        labels_discriminator_real = discriminator(real_data, conditions)

        if par('sa_weights'):
            total_weight = lambda_adv + lambda_mse
            normalized_lambda_adv = lambda_adv / total_weight
            normalized_lambda_mse = lambda_mse / total_weight

        if lambda_adv > 0:
            d_loss = _compute_d_loss(labels_discriminator_fake, labels_discriminator_real, use_wgan=use_wgan, device=device)
            
            if epoch >= epoch_start_training_D:
                d_loss.backward()
                optimizer_D.step()            
        
        optimizer_G.zero_grad()
        optimizer_w.zero_grad()

        labels_discriminator_fake = discriminator(generated_data, conditions)
        g_loss, g_adv, g_mse, g_fft_mse, g_fft_wass = _compute_g_loss(generated_data, real_data, 
                                                                      labels_discriminator_fake, labels_valid, 
                                                                      normalized_lambda_adv, normalized_lambda_mse, 
                                                                      lambda_fftwass, lambda_fftmse,
                                                                      use_wgan=use_wgan, device=device)
        if epoch >= epoch_start_training_G:
            g_loss.backward()
            optimizer_G.step()
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
        torch.save(generator.state_dict(), os.path.join(cp_dir, f"cp-{epoch:04d}.pth"))
        if par('save_discr'):
            torch.save(discriminator.state_dict(), os.path.join(cp_dir, f"cp_discr-{epoch:04d}.pth"))
        print(f"Checkpoint saved at epoch {epoch}")

# Plot the losses over epochs
plt.figure(figsize=(10, 5))
plt.plot(np.array(d_loss_list), label='Discriminator loss')
plt.plot(np.array(g_loss_list), label='Generator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{output_folder}/plots/losses_per_epoch.pdf')

logs.write(log_functions.print_time("Program ended"))
logs.close()