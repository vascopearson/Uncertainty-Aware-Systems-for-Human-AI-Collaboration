import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Coupling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Coupling, self).__init__()
        output_dim = hidden_dim
        
        # Define the scale (s) network
        self.s_fc1 = nn.Linear(input_dim, output_dim)
        self.s_fc2 = nn.Linear(output_dim, output_dim)
        self.s_fc3 = nn.Linear(output_dim, output_dim)
        self.s_fc4 = nn.Linear(output_dim, output_dim)
        self.s_fc5 = nn.Linear(output_dim, input_dim)
        
        # Define the translation (t) network
        self.t_fc1 = nn.Linear(input_dim, output_dim)
        self.t_fc2 = nn.Linear(output_dim, output_dim)
        self.t_fc3 = nn.Linear(output_dim, output_dim)
        self.t_fc4 = nn.Linear(output_dim, output_dim)
        self.t_fc5 = nn.Linear(output_dim, input_dim)

    def forward(self, x):
        # Calculate scale (s) and translation (t)
        s = torch.tanh(self.s_fc5(F.relu(self.s_fc4(F.relu(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x))))))))))
        t = self.t_fc5(F.relu(self.t_fc4(F.relu(self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x)))))))))

        return s, t


class RealNVP(nn.Module):
    def __init__(self, num_coupling_layers, input_dim, hidden_dim=128):
        super(RealNVP, self).__init__()
        self.num_coupling_layers = num_coupling_layers
        self.distribution = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        self.masks = torch.tensor([np.concatenate((np.zeros(input_dim // 2), np.ones(input_dim // 2))), 
                                   np.concatenate((np.ones(input_dim // 2), np.zeros(input_dim // 2)))] * (num_coupling_layers // 2), dtype=torch.float32)
        self.layers_list = nn.ModuleList([Coupling(input_dim, hidden_dim) for _ in range(num_coupling_layers)])

    def forward(self, x, training=True):
        log_det_inv = 0
        direction = -1 if training else 1

        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s = s * reversed_mask
            t = t * reversed_mask
            gate = (direction - 1) / 2

            x = reversed_mask * (x * torch.exp(direction * s) + direction * t * torch.exp(gate * s)) + x_masked
            log_det_inv += gate * torch.sum(s, dim=1)

        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self.forward(x, training=True)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -log_likelihood.mean()

    def score_samples(self, x):
        y, logdet = self.forward(x, training=True)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return log_likelihood
    
