import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import MessagePassing


class GRUCellEx(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCellEx, self).__init__()
        self.gru = nn.GRUCell(input_size, hidden_size, bias=bias)
        self.ingate_layer = nn.Linear(hidden_size, input_size, bias=bias)
        self.ln_i = nn.LayerNorm(input_size)
        self.ln_h = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden):
        gate = torch.sigmoid(self.ingate_layer(hidden))
        gated_input = gate * input
        
        gated_input = self.ln_i(gated_input)
        hidden = self.ln_h(hidden)
        
        return self.gru(gated_input, hidden)


class EdgeConditionedConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.fnet = nn.Sequential(
            nn.BatchNorm1d(edge_dim),
            nn.Linear(edge_dim, 32), 
            nn.ReLU(),
            nn.Linear(32, 128), 
            nn.ReLU(),
            nn.Linear(128, in_channels * out_channels)
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # Orthogonal Init
        for m in self.fnet.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0)
        init.constant_(self.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        weights = self.fnet(edge_attr).view(-1, self.in_channels, self.out_channels)
        # (E, 1, in) @ (E, in, out) -> (E, out)
        return torch.bmm(x_j.unsqueeze(1), weights).squeeze(1)

    def update(self, aggr_out):
        return aggr_out + self.bias


class SPG_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, edge_dim=13, iterations=10):
        super().__init__()
        self.iterations = iterations
        
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        self.ecc = EdgeConditionedConv(hidden_channels, hidden_channels, edge_dim)
        self.gru = GRUCellEx(hidden_channels, hidden_channels)
        
        # Input dimension = hidden * (iterations + 1)
        self.classifier = nn.Linear(hidden_channels * (iterations + 1), num_classes)

    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x)
        history = [h]
        
        for _ in range(self.iterations):
            msg = self.ecc(h, edge_index, edge_attr)
            h = self.gru(msg, h)
            history.append(h)
            
        h_cat = torch.cat(history, dim=1)
        return self.classifier(h_cat)
