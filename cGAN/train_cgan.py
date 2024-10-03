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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Pass YAML configuration to the script.")
parser.add_argument('-p', '--params', required=True, help='Path to the YAML parameters file')
args = parser.parse_args()
with open(os.path.join(script_dir, args.params), 'r') as file:
    params = dict(yaml.safe_load(file))

output_folder = f"{params.get('save_model_dir')}/cgan_{timestamp_now()}"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

logs_path = os.path.join(output_folder, "log.out")
logs = open(logs_path, 'w')
logs.write(log_functions.print_time("Program started"))

with open(os.path.join(output_folder, 'params_model.yaml'), 'w') as file:
    yaml.dump(params, file, default_flow_style=False)

# Load data
if isinstance(params.get('datasets'), list):
    X_list, Y_list = [], []
    for data_dir in params.get('datasets'):
        X, Y, scaler_X, scaler_y, columns_X, columns_Y, info, info_columns = load_preprocessed_data(data_dir, add_info_to_X=params.get('add_info_to_X'))
        X, Y, scaler_X, scaler_y, info = preprocess_data_for_model(X, Y, scaler_X, scaler_y, col_names_X=columns_X, col_names_Y=columns_Y, info=info, params=params)
        X_list.append(X)
        Y_list.append(Y)
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
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
generator = md.Generator(input_features=X.shape[2], hidden_dim=params.get('hidden_dim'), noise_dim=params.get('noise_vector_dim'), 
                         timesteps=params.get('t_len'))

# Load weights of pretrained generator
if (params.get('load_generator_path') is not None) and (params.get('load_generator_cp') is not None):
    weigths_path = os.path.join(params.get('load_generator_path'), "cp", f"cp-{params.get('load_generator_cp'):04d}.pth")
    generator.load_state_dict(torch.load(weigths_path, map_location=device))

# Discriminator
# TODO: This can load one model with different parameters.
if params.get('skip_connections'):
    if params.get('discriminator_input') == "ts":
        discriminator = md.DiscriminatorWithSkipConnections(input_channels, condition_dim)
    elif params.get('discriminator_input') == "fft":
        discriminator = md.DiscriminatorFFTInputWithSkipConnections(input_channels, condition_dim, use_log=params.get('discriminator_use_log'))
else:
    if params.get('discriminator_input') == "ts":
        discriminator = md.Discriminator(input_channels, condition_dim)
    elif params.get('discriminator_input') == "fft":
        discriminator = md.DiscriminatorFFTInput(input_channels, condition_dim, use_log=params.get('discriminator_use_log'))

# Load weights of pretrained discriminator
if (params.get('load_discriminator_path') is not None) and (params.get('load_discriminator_cp') is not None):
    weigths_path = os.path.join(params.get('load_discriminator_path'), "cp", f"cp_discr-{params.get('load_discriminator_cp'):04d}.pth")
    discriminator.load_state_dict(torch.load(weigths_path, map_location=device))



# Optimizers
optimizer_G = get_optimizer(params.get("optimizer_G"), generator, params.get('lr_G'))
optimizer_D = get_optimizer(params.get("optimizer_D"), discriminator, params.get('lr_D'))



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
dataloader = DataLoader(TensorDataset(X, Y), batch_size=params.get('batch_size'), shuffle=True)
nr_batches = len(dataloader)

if not os.path.exists(os.path.join(output_folder, "plots")):
    os.makedirs(os.path.join(output_folder, "plots"))
if not os.path.exists(cp_dir):
    os.makedirs(cp_dir)


# Training
d_loss_list, g_loss_list = [], []
fft_wass_loss, fft_mse = None, None
epoch_start_training_D, epoch_start_training_G = 0, 0

if (params.get('load_discriminator_path') is not None) and (params.get('load_discriminator_cp') is not None):
    epoch_start_training_D = params.get('discriminator_nr_freeze_epochs')
if (params.get('load_generator_path') is not None) and (params.get('load_generator_cp') is not None):
    epoch_start_training_G = params.get('generator_nr_freeze_epochs')

    
for epoch in range(1, params.get('n_epochs')+1):
    t1 = timer()

    for i, (conditions, real_data) in enumerate(dataloader):
        real_data = real_data.to(device).float()
        conditions = conditions.to(device).float()
        current_batch_size = real_data.size(0)

        # Adversarial labels
        valid = torch.ones(current_batch_size, 1, requires_grad=False).to(device)
        fake = torch.zeros(current_batch_size, 1, requires_grad=False).to(device)

        optimizer_D.zero_grad()
        generated_data = generator(conditions)

        
        if params.get('lambda_adversarial', 0) > 0:
            if params.get('use_wgan'):
                d_loss = discriminator(generated_data.detach(), conditions).mean() - discriminator(real_data, conditions).mean()
                gradient_penalty = compute_gradient_penalty(discriminator, real_data, generated_data.detach(), conditions, device)
                d_loss += params.get('lambda_gp') * gradient_penalty
                
            else:
                real_loss = adv_loss(discriminator(real_data, conditions), valid)
                fake_loss = adv_loss(discriminator(generated_data.detach(), conditions), fake)
                d_loss = (real_loss + fake_loss) / 2
            

            if epoch >= epoch_start_training_D:
                d_loss.backward()
                optimizer_D.step()
        else:
            d_loss = torch.tensor(0, device=device, dtype=torch.float32)
        
        d_loss_list.append(d_loss.item())
        optimizer_G.zero_grad()

        g_loss = torch.tensor(0, device=device, dtype=torch.float32)
        g_mse = nn.MSELoss()(generated_data, real_data)

        if params.get('use_wgan'):
            g_loss += -discriminator(generated_data, conditions).mean()

        elif params.get('lambda_adversarial', 0) > 0:
            g_loss += params.get('lambda_adversarial') * adv_loss(discriminator(generated_data, conditions), valid)

        if params.get('lambda_mse', 0) > 0:
            g_loss += params.get('lambda_mse') * g_mse

        elif params.get('lambda_fftwass', 0) > 0:
            fft_wass_loss = losses.FFTWassersteinLoss()(generated_data, real_data)
            g_loss += params.get('lambda_fftwass') * fft_wass_loss

        elif params.get('lambda_fftmse', 0) > 0:
            fft_mse = losses.FFTMSELoss()(generated_data, real_data)
            g_loss += params.get('lambda_fftmse') * fft_mse

        if epoch >= epoch_start_training_G:
            g_loss.backward()
            optimizer_G.step()
        
        g_loss_list.append(g_loss.item())

    t2 = timer()

    if epoch % params.get('log_freq')==0:
        log_out_string = produce_training_log(epoch, t1, t2, d_loss, g_loss, g_mse, fft_wass_loss, fft_mse, params)
        logs.write(log_out_string + '\n')
        print(log_out_string)

    if epoch % params.get('plot_freq')==0:
        plot_gan_samples(real_data[:8].detach().cpu().numpy(), generated_data[:8].detach().cpu().numpy(), num_pairs=8, figsize=(5, 5), 
                            plot_name=f'comparison_output_{epoch}',
                            output_folder=os.path.join(output_folder, "plots"))
            
    if epoch % params.get('cp_freq')==0:
        torch.save(generator.state_dict(), os.path.join(cp_dir, f"cp-{epoch:04d}.pth"))
        if params.get('save_discr'):
            torch.save(discriminator.state_dict(), os.path.join(cp_dir, f"cp_discr-{epoch:04d}.pth"))
        print(f"Checkpoint saved at epoch {epoch}")

# Plot the losses over iterations
plt.figure(figsize=(10, 5))
plt.plot(d_loss_list, label='Discriminator loss')
plt.plot(g_loss_list, label='Generator loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{output_folder}/plots/losses_per_iteration.pdf')

# Plot the losses over epochs
plt.figure(figsize=(10, 5))
plt.plot(np.array(d_loss_list).reshape(-1, nr_batches).mean(axis=1), label='Discriminator loss')
plt.plot(np.array(g_loss_list).reshape(-1, nr_batches).mean(axis=1), label='Generator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{output_folder}/plots/losses_per_epoch.pdf')

logs.write(log_functions.print_time("Program ended"))
logs.close()