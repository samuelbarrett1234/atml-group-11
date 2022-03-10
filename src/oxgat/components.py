import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class GATLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 leaky_relu_slope: float = 0.2,
                 num_heads: int = 1,
                 is_final_layer: bool = False):
        super(GATLayer, self).__init__()
        self.is_final_layer = is_final_layer
        self.heads = torch.nn.ModuleList([AttentionHead(in_features,
                                                        out_features,
                                                        leaky_relu_slope)
                                          for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        outputs = [head(x, edge_index) for head in self.heads]
        output = torch.mean(torch.stack(outputs)) \
                 if self.is_final_layer \
                 else torch.concat(outputs, dim=2)
        return F.elu(output)


class AttentionHead(torch.nn.Module):
    def __init__(self, in_features, out_features, alpha):
        super(AttentionHead, self).__init__()

        self.W = torch.nn.Linear(in_features, out_features, bias=False)
        self.a = torch.nn.Linear(out_features*2, 1, bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.W(x) # Shared linear map
        attention = self._attention(x, to_dense_adj(edge_index))
        return attention @ x

    def _attention(self, x, adj):
        n = x.shape[0]

        # Get attention mechanism input for each pair of nodes
        x_repeated = x.repeat(1,1,n).transpose(1,2) # Shape (n,n,F')
        feature_pairs = torch.cat([x_repeated,
                                   x_repeated.transpose(0,1)],
                                  dim=2) # Shape (n,n,2F')

        # Calculate attention for each edge
        e = torch.where(adj > 0, self.a(feature_pairs), torch.zeros(n,n))
        attention = F.softmax(self.leaky_relu(e), dim=1)

        return attention