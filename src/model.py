import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, hidden_dim):
        super().__init__()

        self.W_e = nn.Linear(encoder_dim, hidden_dim)
        self.W_d = nn.Linear(decoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, e, d):
        """
        e: encoder features (skip)      [batch, encoder_dim]
        d: decoder features (current)  [batch, decoder_dim]
        """

        # project into shared space
        e_proj = self.W_e(e)
        d_proj = self.W_d(d)

        # combine
        attn = self.v(torch.tanh(e_proj + d_proj)) 
        attn = self.sigmoid(attn)

        # gate encoder features
        return e * attn

class VAE(nn.Module):
    def __init__(
        self,
        input_dim = 109792,
        hidden_dims=[8192, 4096, 1024, 256],
        latent_dim=128,
        dropout=0.1,
        decode_alpha=1
    ):
        """
        Args:
            input_dim (int): length of input vector
            hidden_dims (str): neural network hidden layer dimensions
            latent_dim (list): latent space dims
        """
        
        super().__init__()

        self.attention_gates = nn.ModuleList()

        # encoder
        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # decoder
        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)


        for e_dim, d_dim in zip(hidden_dims[::-1], hidden_dims):
            self.attention_gates.append(
                AttentionGate(e_dim, d_dim, hidden_dim=d_dim // 2)
            )

    def encode(self, x):
        activations = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):  
                activations.append(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar, activations

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, activations):
        gate_idx = 0
        for i, layer in enumerate(self.decoder[:-1]):
            z = layer(z)

            if i < len(activations):
                skip = activations[-(i+1)]
                gated_skip = self.attention_gates[i](skip, z)

                z = z + decode_alpha * gated_skip

        return self.decoder[-1](z)

    def forward(self, x):
        mu, logvar, activations = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, activations)

        return x_hat, mu, logvar