import torch
import torch.nn as nn


class DAE(nn.Module):
    def __init__(
        self,
        input_dim = 109792,
        hidden_dims=[1024, 256],
        latent_dim=64
    ):
        """
        Args:
            input_dim (int): length of input vector
            hidden_dims (str): neural network hidden layer dimensions
            latent_dim (list): latent space dims
        """
        
        super().__init__()

        # encoder
        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_latent=False):
        z = self.encode(x)
        x_hat = self.decode(z)
    
        if return_latent:
            return x_hat, z
    
        return x_hat