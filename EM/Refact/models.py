import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    device = "cpu"

from torch.distributions import Normal 


class AffineCouplingLayer(nn.Module):
    def __init__(self, num_features, hidden_dim, memory_size):
        super(AffineCouplingLayer, self).__init__()
        
        self.net = CustomNet(num_features=num_features, hidden_dim=hidden_dim)
        self.memory_size = memory_size
    def forward(self, x):
        mask = torch.zeros(size = (1, self.memory_size, 1))
        mask[:, :self.memory_size//2, :] = 1
        x1 = x*mask
        x2 = x*(1-mask)
        params = self.net(x1)

        shift, log_scale = params.chunk(2, dim=-1)
        shift = shift*(1-mask)
        log_scale = log_scale*(1-mask)

        z = (x + shift)*torch.exp(log_scale)
        log_det_jacobian = log_scale.sum(dim=(1,2))
        
        
        return z, log_det_jacobian

    def inverse(self, z):
        raise NotImplemented
        # z1, z2 = z.chunk(2, dim=1)
        # params = self.net(z1)
        # shift, log_scale = params.chunk(2, dim=1)
        # x2 = z2 * torch.exp(log_scale) + shift
        # x = torch.cat([z1, x2], dim=1)
        
        # return x

class InvConv1D(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Initialize an orthogonal matrix for 1x1 convolution in 1D
        w_init, _ = torch.linalg.qr(torch.randn(num_features, num_features))
        #w_init = torch.tensor(w_init)

        self.register_buffer('w', w_init)
        self.w_inverse = None

    def forward(self, x):
        # x is of shape (batch_size, num_features)
        # Apply the learned 1x1 convolution (linear transformation)
        z = self.w @ x  # Matrix multiplication for 1D

        # Compute log determinant of the Jacobian for the flow
        log_det_jacobian = torch.slogdet(self.w)[1]
        
        return z, log_det_jacobian

    def inverse(self, z):
        if self.w_inverse is None:
            self.w_inverse = torch.inverse(self.w)
        x = self.w_inverse @ z
        
        return x



class NFBlock(nn.Module):
    def __init__(self, num_features, hidden_dim, memory_size):
        super().__init__()
        self.invertible_conv = InvConv1D(memory_size)
        self.affine_coupling = AffineCouplingLayer(num_features = num_features, hidden_dim = hidden_dim, memory_size = memory_size)

    def forward(self, x):
        z, log_det_jacobian_conv = self.invertible_conv(x)
        
        z, log_det_jacobian_affine = self.affine_coupling(z)
        
        log_det_jacobian = log_det_jacobian_conv + log_det_jacobian_affine
        
        return z, log_det_jacobian

    def inverse(self, z):
        x = self.affine_coupling.inverse(z)
        x = self.invertible_conv.inverse(x)
        
        return x

class CustomNet(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        
        self.embd = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.Tanh())
        
        self.rnn = nn.GRU(input_size = hidden_dim, hidden_size = int(2*hidden_dim), num_layers = 2, batch_first = True)
        self.fc = nn.Linear(int(2*hidden_dim), 2*num_features)

    def forward(self, x):
        
        
        x = self.embd(x) #(10, 4, 1) -> (10, 4, 32)
        
        x, _ = self.rnn(x) # (10, 4, 32) -> (10, 4, 64)
        

        x = self.fc(x) # (10, 4, 32) -> (10, 4, 2)

        return x



class NormalizingFlow(nn.Module):
    def __init__(self, num_features, memory_size, hidden_dim, num_layers):
        super(NormalizingFlow, self).__init__()
        self.num_features = num_features
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.base_dist = Normal(0, 1)
        self.blocks = nn.ModuleList(
            [NFBlock(num_features = num_features, memory_size = memory_size, hidden_dim = hidden_dim) for _ in range(num_layers)]
        )
        
    
    def forward(self, x):
        log_det_jacobian = 0
        
        for layer in self.blocks:
            x, ldj = layer(x)
            log_det_jacobian += ldj
        
        return x, log_det_jacobian

    def inverse(self, z):
        for layer in reversed(self.blocks):
            z = layer.inverse(z)
        return z


    def log_prob(self, x):
        z, ldj = self.forward(x)
        log_pz = self.base_dist.log_prob(z).sum(dim = (1,2))
        log_px = log_pz + ldj

        return z, log_px

