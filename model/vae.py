import torch
import math
import torch.nn as nn
import numpy as np
import torchvision.models as models

r"""
A very simple VAE which has the following architecture
Encoder
    N * Conv BN Activation Blocks
    FC layers for mean
    FC layers for variance

Decoder
    FC Layers taking z to higher dimensional feature
    N * ConvTranspose BN Activation Blocks
"""
class VAE(nn.Module):
    def __init__(self,
                 config
                 ):
        super(VAE, self).__init__()
        activation_map = {
            'relu':nn.ReLU(),
            'leaky':nn.LeakyReLU(),
            'tanh':nn.Tanh(),
            'gelu':nn.GELU(),
            'silu':nn.SiLU()
        }
        
        self.config = config
        ##### Validate the configuration for the model is correctly setup #######
        assert config['transpose_activation_fn'] is None or config['transpose_activation_fn'] in activation_map
        assert config['dec_fc_activation_fn'] is None or config['dec_fc_activation_fn'] in activation_map
        assert config['conv_activation_fn'] is None or config['conv_activation_fn'] in activation_map
        assert config['enc_fc_activation_fn'] is None or config['enc_fc_activation_fn'] in activation_map
        assert config['enc_fc_layers'][-1] == config['dec_fc_layers'][0] == config['latent_dim'], \
            "Latent dimension must be same as fc layers number"
        
        
        self.transposebn_channels = config['transposebn_channels']
        self.latent_dim = config['latent_dim']
        
        # Encoder is just Conv bn blocks followed by fc for mean and variance
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(config['convbn_channels'][i], config['convbn_channels'][i+1],
                          kernel_size=config['conv_kernel_size'][i], stride=config['conv_kernel_strides'][i]),
                nn.BatchNorm2d(config['convbn_channels'][i+1]),
                activation_map[config['conv_activation_fn']]
            )
            for i in range(config['convbn_blocks'])
        ])
        encoder_mu_activation = nn.Identity() if config['enc_fc_mu_activation'] is None else activation_map[config['enc_fc_mu_activation']]
        self.encoder_mu_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['enc_fc_layers'][i], config['enc_fc_layers'][i+1]),
                encoder_mu_activation
            )
            for i in range(len(config['enc_fc_layers'])-1)
        ])
        encoder_var_activation = nn.Identity() if config['enc_fc_var_activation'] is None else activation_map[config['enc_fc_var_activation']]
        self.encoder_var_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['enc_fc_layers'][i], config['enc_fc_layers'][i + 1]),
                encoder_var_activation
            )
            for i in range(len(config['enc_fc_layers']) - 1)
        ])
        
        # Decoder is just fc followed by Convtranspose bn blocks
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(config['transposebn_channels'][i], config['transposebn_channels'][i + 1], kernel_size=config['transpose_kernel_size'][i],
                          stride=config['transpose_kernel_strides'][i]),
                nn.BatchNorm2d(config['transposebn_channels'][i + 1]),
                activation_map[config['transpose_activation_fn']]
            )
            for i in range(config['transpose_bn_blocks'])
        ])
        self.decoder_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['dec_fc_layers'][i], config['dec_fc_layers'][i+1]),
                activation_map[config['dec_fc_activation_fn']]
            )
            for i in range(len(config['dec_fc_layers'])-1)
            
        ])
        
    def forward(self, x):
        out = x
        for layer in self.encoder_layers:
            out = layer(out)
        out = out.reshape((x.size(0),-1))
        mu = out
        for layer in self.encoder_mu_fc:
            mu = layer(mu)
        std = out
        for layer in self.encoder_var_fc:
            std = layer(std)
        z = self.reparameterize(mu, std)
        generated_out = self.generate(z)
        if self.config['log_variance']:
            return {
                'mean':mu,
                'log_variance':std,
                'image':generated_out,
            }
        else:
            return {
                'mean': mu,
                'std': std,
                'image': generated_out,
            }
        
    
    def generate(self, z):
        out = z
        for layer in self.decoder_fc:
            out = layer(out)
        hw = out.shape[-1] / self.transposebn_channels[0]
        spatial = int(math.sqrt(hw))
        assert spatial*spatial == hw
        out = out.reshape((z.size(0), -1, spatial, spatial))
        for layer in self.decoder_layers:
            out = layer(out)
        return out
    
    def sample(self, num_images=1, z=None):
        if z is None:
            z = torch.randn((num_images, self.enc_fc_layers[-1]))
        assert z.size(0) == num_images
        out = self.generate(z)
        return out
        
    def reparameterize(self, mu, std_or_logvariance):
        if self.config['log_variance']:
            std = torch.exp(0.5 * std_or_logvariance)
        else:
            std = std_or_logvariance
        z = torch.randn_like(std)
        return z * std + mu
        
    
def get_model(config):
    model = VAE(
        config=config['model_params']
    )
    return model

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import yaml
    
    config_path = '../config/vae_nokl.yaml'
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    model = get_model(config)
    
    out = model(torch.rand(1,1,28,28))
    print(out.keys())
