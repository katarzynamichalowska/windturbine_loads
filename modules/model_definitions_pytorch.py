import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class Model1(nn.Module):
    def __init__(self, input_features, sequence_length, output_features, t_len):
        super(Model1, self).__init__()
        self.t_len = t_len
        self.output_features = output_features
        # Regularization with weight norm
        self.dense1 = weight_norm(nn.Linear(input_features, 30))
        #self.bn1 = nn.BatchNorm1d(sequence_length)  # Apply BatchNorm on the sequence dimension
        self.lstm = nn.LSTM(input_size=30, hidden_size=30, batch_first=True)
        self.dense2 = weight_norm(nn.Linear(30, 30))
        self.dropout = nn.Dropout(0.2)
        self.dense3 = weight_norm(nn.Linear(30, 10))
        self.dense4 = weight_norm(nn.Linear(10, 3))
        self.flatten = nn.Flatten()
        self.dense5 = weight_norm(nn.Linear(3 * sequence_length, t_len * output_features))
        self.final_reshape = nn.Unflatten(1, (t_len, output_features))

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        #x = x.permute(0, 2, 1)
        #x = self.bn1(x)
        #x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout(x)
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = self.flatten(x)
        x = self.dense5(x)
        x = self.final_reshape(x)
        return x


class GeneratorLSTM(nn.Module):
    def __init__(self, input_features, sequence_length, output_features, t_len, noise_dim):
        super(GeneratorLSTM, self).__init__()
        self.t_len = t_len
        self.output_features = output_features
        self.noise_dim = noise_dim
        
        # Updated input feature size to include noise
        total_input_features = input_features + noise_dim
        
        # Regularization with weight norm
        self.dense1 = weight_norm(nn.Linear(total_input_features, 30))
        self.lstm = nn.LSTM(input_size=30, hidden_size=30, batch_first=True)
        self.dense2 = weight_norm(nn.Linear(30, 30))
        self.dropout = nn.Dropout(0.2)
        self.dense3 = weight_norm(nn.Linear(30, 10))
        self.dense4 = weight_norm(nn.Linear(10, 3))
        self.flatten = nn.Flatten()
        self.dense5 = weight_norm(nn.Linear(3 * sequence_length, t_len * output_features))
        self.final_reshape = nn.Unflatten(1, (t_len, output_features))

    def forward(self, x):
        # Check that noise is the correct shape
        #assert noise.shape[1] == self.noise_dim, "Noise dimension mismatch"
        #assert x.shape[0] == noise.shape[0], "Batch size of input and noise must match"

        # Concatenate input with noise
        #x = torch.cat((x, noise), dim=1)
        
        x = torch.relu(self.dense1(x))
        x, _ = self.lstm(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout(x)
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = self.flatten(x)
        x = self.dense5(x)
        x = self.final_reshape(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, input_features, hidden_dim, noise_dim, output_dim=1, timesteps=300):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.timesteps = timesteps
        
        # LSTM layer that will handle the time series and noise input
        self.lstm = nn.LSTM(input_size=input_features + noise_dim, 
                            hidden_size=hidden_dim,
                            batch_first=True)
        
        # Additional dense layers
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.output_activation = nn.Linear(output_dim,output_dim)

    def forward(self, x):
        # Generating noise
        batch_size = x.size(0)
        # Directly generate noise for all timesteps and batch
        noise = torch.randn(batch_size, self.timesteps, self.noise_dim, device=x.device)
        
        # Concatenate noise to the input features at each timestep
        combined_input = torch.cat((x, noise), dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined_input)
        #lstm_out, _ = self.lstm(x)

        # Dense layers processing
        x = self.tanh(self.dense1(lstm_out))
        output = self.tanh(self.dense2(x))
        
        # Output activation (e.g., tanh for normalized output)
        output = self.output_activation(output)
        
        return output


class Discriminator(nn.Module):
    def __init__(self, input_channels, condition_dim, output_channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Assume input has been permuted to (batch_size, channels, sequence_length)
            nn.Conv1d(in_channels=input_channels + condition_dim, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, output_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),  # Reduce to one feature per example
            nn.Flatten(),  # Flatten to match expected output shape for loss calculation
            nn.Sigmoid()  # Ensure output is a valid probability
        )

    def forward(self, data, conditions):
        # Assume data and conditions are in shape (batch_size, sequence_length, features/channels)
        # Concatenate data and conditions along the feature/channel dimension
        combined_input = torch.cat((data, conditions), dim=2)  # Resulting shape (batch_size, sequence_length, input_channels + condition_dim)
        # Permute to (batch_size, channels, sequence_length) to fit Conv1d requirement
        combined_input = combined_input.permute(0, 2, 1)
        return self.model(combined_input)
    

import torch
import torch.nn as nn

class GeneratorFourier(nn.Module):
    def __init__(self, input_features, hidden_dim, noise_dim, output_dim=1, timesteps=300):
        super(GeneratorFourier, self).__init__()
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.timesteps = timesteps
        
        # LSTM layer that will handle the time series and noise input
        self.lstm = nn.LSTM(input_size=input_features + noise_dim, 
                            hidden_size=hidden_dim,
                            batch_first=True)
        
        # Fourier layer
        self.fourier_layer = FourierLayer(input_features + noise_dim)
        
        # Additional dense layers
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation functions
        self.tanh = nn.Tanh()
        self.output_activation = nn.Linear(hidden_dim, output_dim) #nn.Tanh()

    def forward(self, x):
        # Generating noise
        batch_size = x.size(0)
        # Directly generate noise for all timesteps and batch
        noise = torch.randn(batch_size, self.timesteps, self.noise_dim, device=x.device)
        
        # Concatenate noise to the input features at each timestep
        combined_input = torch.cat((x, noise), dim=-1)
        
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined_input)

        # Skip connection: concatenate Fourier output with LSTM output
        #combined_output = torch.cat((lstm_out, fourier_output), dim=-1)

        # Dense layers processing
        x = self.tanh(self.dense1(lstm_out))
        output = self.dense2(x)

        # Fourier layer processing
        fourier_output = self.fourier_layer(output)

        
        # Skip connection: add Fourier output to the dense layer output
        output += fourier_output
        
        # Output activation (e.g., tanh for normalized output)
        #output = self.output_activation(output)
        
        return output


#class FourierLayer(nn.Module):
#    def __init__(self, input_dim):
#        super(FourierLayer, self).__init__()
#        self.input_dim = input_dim

##    def forward(self, x):
 #       # Assuming x is of shape (batch_size, timesteps, input_dim)
 #       # Perform Fourier transform along the input dimension
 #       fourier_transform = torch.fft.fft(x, dim=-1)
 #       return fourier_transform

class FourierLayer(nn.Module):
    def __init__(self, input_dim):
        super(FourierLayer, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        # Assuming x is of shape (batch_size, timesteps, input_dim)
        # Convert input to complex data type
        #x_complex = torch.view_as_complex(x)

        # Perform Fourier transform along the input dimension
        fourier_transform = torch.abs(torch.fft.rfft2(x)) # Only the amplitudes of the spectra

        #fourier_transform = torch.fft.fft(x_complex, dim=-1)
        return fourier_transform
    
class WassersteinLoss(nn.Module):
    def __init__(self, epsilon=0.01, n_iter=50):
        super(WassersteinLoss, self).__init__()
        self.epsilon = epsilon
        self.n_iter = n_iter

    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        distances = []
        
        for i in range(batch_size):
            # Extract individual samples
            pred_sample = y_pred[i]
            true_sample = y_true[i]
            
            # Compute pairwise distance matrix
            C = torch.cdist(pred_sample.unsqueeze(0), true_sample.unsqueeze(0), p=2).squeeze()
            
            # Initialize the dual variables
            u = torch.zeros(pred_sample.shape[0], dtype=torch.float32, requires_grad=False).to(pred_sample.device)
            v = torch.zeros(true_sample.shape[0], dtype=torch.float32, requires_grad=False).to(true_sample.device)

            # Sinkhorn iterations
            for _ in range(self.n_iter):
                u = self.epsilon * (torch.logsumexp((v - C).div(self.epsilon), dim=1)) + u
                v = self.epsilon * (torch.logsumexp((u - C.t()).div(self.epsilon), dim=1)) + v

            # Compute transport plan
            transport_plan = torch.exp((u[:, None] + v[None, :] - C) / self.epsilon)
            # Compute Wasserstein distance
            wasserstein_distance = torch.sum(transport_plan * C)
            
            distances.append(wasserstein_distance)
        
        # Return mean distance over the batch
        return torch.mean(torch.stack(distances))

class WassersteinLoss(nn.Module):
    def __init__(self, epsilon=0.01, n_iter=50):
        super(WassersteinLoss, self).__init__()
        self.epsilon = epsilon
        self.n_iter = n_iter

    def forward(self, y_pred, y_true):
        batch_size, seq_length,_ = y_pred.size()
        
        # Compute pairwise distance matrix for the batch
        C = torch.cdist(y_pred, y_true, p=2)  # Shape: [batch_size, seq_length, seq_length]
        
        # Initialize the dual variables
        u = torch.zeros(batch_size, seq_length, dtype=torch.float32, requires_grad=False).to(y_pred.device)
        v = torch.zeros(batch_size, seq_length, dtype=torch.float32, requires_grad=False).to(y_true.device)

        # Sinkhorn iterations
        for _ in range(self.n_iter):
            u = self.epsilon * (torch.logsumexp((v[:, None, :] - C).div(self.epsilon), dim=2)) + u
            v = self.epsilon * (torch.logsumexp((u[:, :, None] - C).div(self.epsilon), dim=1)) + v

        # Compute transport plan
        transport_plan = torch.exp((u[:, :, None] + v[:, None, :] - C) / self.epsilon)
        
        # Compute Wasserstein distance
        wasserstein_distance = torch.sum(transport_plan * C, dim=[1, 2])
        
        return torch.mean(wasserstein_distance)

