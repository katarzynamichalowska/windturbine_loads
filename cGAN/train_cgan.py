import os
import yaml
import numpy as np
import sys
sys.path.insert(0, '/home/katarzynam/windturbine_loads')
from modules.dir_functions import timestamp_now
from modules.data_manipulation import preprocess_data_for_model
from modules.data_loading import load_preprocessed_data
from modules import log_functions
from modules.plotting import plot_gan_samples
import modules.model_definitions_pytorch as md
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('/home/katarzynam/windturbine_loads/cGAN/params_train.yaml', 'r') as file:
    params = dict(yaml.safe_load(file))

main_model_folder = params.get('model_dir')
output_folder = f"{main_model_folder}/cgan_{timestamp_now()}"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'params_model.yaml'), 'w') as file:
    yaml.dump(params, file, default_flow_style=False)

# Params
if isinstance(params.get('data_seed'), list):
    X_list, Y_list = [], []
    for i in params.get('data_seed'):
        data_dir = f"/home/katarzynam/data/data_X_y_scaled_seed_{i}.npz"
        X, Y, scaler_X, scaler_y, columns_X, columns_Y, info, info_columns = load_preprocessed_data(data_dir, add_info_to_X=params["add_info_to_X"])
        X, Y, scaler_X, scaler_y, info = preprocess_data_for_model(X, Y, scaler_X, scaler_y, col_names_X=columns_X, col_names_Y=columns_Y, info=info, params=params)


        X_list.append(X)
        Y_list.append(Y)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

variance_y = np.var(Y)

checkpoint_dir = os.path.join(output_folder, "cp")

logs_path = os.path.join(output_folder, "log.out")
logs = open(logs_path, 'w')
logs.write(log_functions.print_time("Program started"))

# Parameters
num_samples = X.shape[0]
timesteps = X.shape[1]
condition_dim = X.shape[2]  # Condition dimension
input_channels = Y.shape[2]  # Number of output channels from the generator, matches real data channels

# Initialize models with dynamic dimensions
#generator = md.Generator(noise_dim, condition_dim)
generator = md.Generator(input_features=X.shape[2], hidden_dim=100, noise_dim=params["noise_vector_dim"], 
                         timesteps=params["t_len"])

if params.get("model_preload") is not None:
    weigths_path = os.path.join(main_model_folder, params["model_preload"], "checkpoints", f"cp-{params['model_preload_cp']:04d}.pth")
    if device==torch.device('cuda'):
        generator.load_state_dict(torch.load(weigths_path))
    else:
        generator.load_state_dict(torch.load(weigths_path, map_location=torch.device('cpu')))

discriminator = md.Discriminator(input_channels, condition_dim)


# Example training setup (assuming conditions and real_data are defined)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=params.get('lr_G'), betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=params.get('lr_D'), betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Device configuration
generator.to(device)
discriminator.to(device)

#print("Generator Summary:")
#summary(generator, input_size=[(timesteps, noise_dim), (timesteps, condition_dim)])
#summary(generator, input_size=[(timesteps, noise_dim), (timesteps, condition_dim)])

#print("\nDiscriminator Summary:")
#summary(discriminator, input_size=[(timesteps, 1), (timesteps, condition_dim)])

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)


# Create dataset and dataloader
print("Training data shapes:")
print(f"X: {X.shape}")
print(f"Y: {Y.shape}")
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
nr_batches = len(dataloader)

if not os.path.exists(os.path.join(output_folder, "plots")):
    os.makedirs(os.path.join(output_folder, "plots"))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Training Loop
d_loss_list = []
g_loss_list = []

    
for epoch in range(1, params.get('n_epochs')+1):
    for i, (conditions, real_data) in enumerate(dataloader):
        # Reshape data
        real_data = real_data.to(device).float()
        conditions = conditions.to(device).float()
        current_batch_size = real_data.size(0)

        # Adversarial labels
        valid = torch.ones(current_batch_size, 1, requires_grad=False).to(device)
        fake = torch.zeros(current_batch_size, 1, requires_grad=False).to(device)

        optimizer_D.zero_grad()
        
        generated_data = generator(conditions)

        real_loss = criterion(discriminator(real_data, conditions), valid)
        fake_loss = criterion(discriminator(generated_data.detach(), conditions), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss_list.append(d_loss.item())

        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(generated_data, conditions), valid)
        g_loss_list.append(g_loss.item())
        #if (params["model_preload"] is not None) and (params["model_preload_cp"] is not None) and (epoch > params["model_preload_cp"]):
        g_loss.backward()

        optimizer_G.step()

        if (epoch % params.get('log_freq')==0) & (i == 0):
            print(f"[Epoch {epoch}/{params["n_epochs"]}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            plot_gan_samples(epoch, real_data[:8], generated_data[:8], num_pairs=8, figsize=(10, 5), 
                             output_folder=os.path.join(output_folder, "plots"))
            
    if epoch % params.get('cp_freq')==0:
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"cp-{epoch:04d}.pth"))
        #torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f"cp-{epoch:04d}.pth"))
        print(f"Checkpoint saved at epoch {epoch}")

# Plot the losses over epochs
plt.figure(figsize=(10, 5))
plt.plot(d_loss_list, label='Discriminator loss')
plt.plot(g_loss_list, label='Generator loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{output_folder}/plots/losses_per_iteration.pdf')

# The same figure but mean for each epoch
plt.figure(figsize=(10, 5))
plt.plot(np.array(d_loss_list).reshape(-1, nr_batches).mean(axis=1), label='Discriminator loss')
plt.plot(np.array(g_loss_list).reshape(-1, nr_batches).mean(axis=1), label='Generator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{output_folder}/plots/losses_per_epoch.pdf')

