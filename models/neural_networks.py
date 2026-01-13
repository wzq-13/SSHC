from torch import nn
import torch
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU()]

        for i in range(len(hidden_dim) - 1):
            layers += [nn.Linear(hidden_dim[i], hidden_dim[i+1]), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout)) # Adjust dropout rate for each layer

        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, data):
        B = data['obstacles_vertices'].shape[0]
        obstacles_vertices = data['obstacles_vertices'].view(B, -1)
        target = data['target'].view(B, -1)
        x = torch.cat([obstacles_vertices, target], dim=1)
        output = self.mlp(x)
        return output.view(B, -1)