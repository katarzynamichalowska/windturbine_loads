import os
import yaml
import numpy as np
import sys
sys.path.insert(0, '/home/katarzynam/windturbine_loads')
from modules.data_manipulation import preprocess_data_for_model
from modules.data_loading import load_preprocessed_data
import modules.model_definitions_pytorch as md
import torch
import matplotlib.pyplot as plt
from modules.plotting import plot_gan_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('/home/katarzynam/windturbine_loads/cGAN/params_test.yaml', 'r') as file:
    params = dict(yaml.safe_load(file))

main_model_folder = params.get('model_dir')
modelname = params.get('model')
output_folder = f"{main_model_folder}/{modelname}"

with open(f'{output_folder}/params_model.yaml', 'r') as file:
    params_model = dict(yaml.safe_load(file))

# Params
if isinstance(params.get('data_seed'), list):
    X_list, Y_list = [], []
    for i in params.get('data_seed'):
        data_dir = f"/home/katarzynam/data/data_X_y_scaled_seed_{i}.npz"
        X, Y, scaler_X, scaler_y, columns_X, columns_Y, info, info_columns = load_preprocessed_data(data_dir, add_info_to_X=params_model.get('add_info_to_X'))
        X, Y, scaler_X, scaler_y, info = preprocess_data_for_model(X, Y, scaler_X, scaler_y, col_names_X=columns_X, col_names_Y=columns_Y, info=info, params=params_model)


        X_list.append(X)
        Y_list.append(Y)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

variance_y = np.var(Y)

checkpoint_dir = os.path.join(output_folder, "cp")

# Parameters
num_samples = X.shape[0]
timesteps = X.shape[1]
condition_dim = X.shape[2]  # Condition dimension
input_channels = Y.shape[2]  # Number of output channels from the generator, matches real data channels

X = torch.tensor(X, dtype=torch.float32)
#Y = torch.tensor(Y, dtype=torch.float32)

# Create dataset and dataloader
print("Testing data shapes:")
print(f"X: {X.shape}")
print(f"Y: {Y.shape}")
#dataset = TensorDataset(X, Y)
#dataloader = DataLoader(dataset, batch_size=params_model.get('batch_size'), shuffle=False)

if not os.path.exists(os.path.join(output_folder, "testing")):
    os.makedirs(os.path.join(output_folder, "testing"))

# Initialize models with dynamic dimensions
#generator = md.Generator(noise_dim, condition_dim)
generator = md.Generator(input_features=X.shape[2], hidden_dim=100, noise_dim=params_model["noise_vector_dim"], 
                         timesteps=params_model.get('t_len'))

generator.to(device)

np.random.seed(42)
indices = np.random.choice(len(Y), 8, replace=False)

mse_loss_list = []
fft_mse_loss_list = []

for epoch in params.get('cp'):
    print(f"Epoch: {epoch}")
    weigths_path = os.path.join(main_model_folder, modelname, "cp", f"cp-{epoch:04d}.pth")
    if device==torch.device('cuda'):
        generator.load_state_dict(torch.load(weigths_path))
    else:
        generator.load_state_dict(torch.load(weigths_path, map_location=torch.device('cpu')))

    Y_gen = generator(X)
    Y_gen = Y_gen.detach().cpu().numpy()

    # Statistics on timeseries
    plot_gan_samples(Y[indices], Y_gen[indices], num_pairs=8, figsize=(3, 5), 
                    plot_name=f'ts_e{epoch}',
                    output_folder=os.path.join(output_folder, "testing"))
    
    mse_loss_list.append(np.mean((Y - Y_gen)**2))

    # Statistics on FFT
    Y_gen_fft = np.abs(np.fft.fft(Y_gen, axis=1))[:, :timesteps//2]
    Y_fft = np.abs(np.fft.fft(Y, axis=1))[:, :timesteps//2]

    fft_mse_loss_list.append(np.mean((Y_fft - Y_gen_fft)**2))
    
    plot_gan_samples(Y_fft[indices], Y_gen_fft[indices], num_pairs=8, figsize=(3, 5), 
                    plot_name=f'fft_e{epoch}',
                    output_folder=os.path.join(output_folder, "testing"))
    plot_gan_samples(Y_fft[indices], Y_gen_fft[indices], num_pairs=8, figsize=(3, 5), 
                    plot_name=f'fft_e{epoch}_log',
                    output_folder=os.path.join(output_folder, "testing"), log=True)


plt.figure(figsize=(5, 4))
plt.plot(params.get('cp'), mse_loss_list)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("MSE Loss per Epoch")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "testing", "ts_mse_loss.pdf"))

plt.figure(figsize=(5, 4))
plt.plot(params.get('cp'), fft_mse_loss_list)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("FFT MSE Loss per Epoch")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "testing", "fft_mse_loss.pdf"))


# Add new losses. Compare on meaningful features of the data.