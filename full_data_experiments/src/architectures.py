import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GINConv, GCNConv




class FCL(nn.Module):
  ''' This network is used in the ablation study to measure the contribute of the GCN layers '''
  def __init__(self):
    super(FCL, self).__init__()
    self.FC1 = nn.Linear(2613, 256)
    self.FC2 = nn.Linear(256, 256)
    self.FC3 = nn.Linear(256, 256)

  def forward(self, x, edges):
    out = F.elu(self.FC1(x))
    out = F.elu(self.FC2(out))
    out = self.FC3(out)

    return out

class GATSYFC(nn.Module):

  def __init__(self, n_heads, n_layers):
    super(GATSYFC, self).__init__()
    self.GATSY = GATSY(n_heads, n_layers)
    self.pred = Predictor(n_heads)

  def forward(self, x, edges):

    x = self.GATSY(x, edges)
    x = self.pred(x)

    return x



class Predictor(nn.Module):
  ''' This architecture is used as head of GAT2, to predict the genres of an artist '''
  def __init__(self, n_heads):
    super(Predictor, self).__init__()
    

    self.linear1 = nn.Linear(n_heads*256,n_heads*64)
    self.linear2 = nn.Linear(n_heads*64,n_heads*64)
    self.linear3 = nn.Linear(n_heads*64,25)
    self.batch1 = torch.nn.BatchNorm1d(n_heads*64)
    self.batch2 = torch.nn.BatchNorm1d(n_heads*64)

  def forward(self, x):

    x = self.linear1(x)
    x = self.batch1(x)
    x = F.elu(x)

    x = self.linear2(x)
    x = self.batch2(x)
    x = F.elu(x)
    x = self.linear3(x)
    
    return x

# class GATSY(nn.Module):
#   ''' This architecture is one of the 2 GAT networks, its aim is to be improved in order to see how it performs on the artist similarity task '''
#   def __init__(self, n_heads, n_layers):
#     super(GATSY, self).__init__()

#     ''' The Batch normalization layers have been introduced to speed the training up, and indeed to obtain better results. '''
    
    
#     self.batch1 = torch.nn.BatchNorm1d(256)
#     self.batch2 = torch.nn.BatchNorm1d(256)
#     self.batch3 = torch.nn.BatchNorm1d(256)
#     self.batch4 = torch.nn.BatchNorm1d(n_heads*256)

#     self.n_layer = n_layers
#     GAT_l = []
#     self.GAT1 = GATConv(256,256, heads = n_heads, bias = True)
#     for n in range(n_layers - 1):
#       GAT_l.append(GATConv(n_heads*256,256, heads = n_heads, bias = True))

#     self.GAT_l = nn.Sequential(*GAT_l)

#     batch_l = []

#     for n in range(n_layers - 1):
#       batch_l.append(torch.nn.BatchNorm1d(n_heads*256))

#     self.batch_l = nn.Sequential(*batch_l)


#     self.linear1 = nn.Linear(2613,256)
#     self.linear2 = nn.Linear(256,256)
#     self.linear3 = nn.Linear(256,256)


  
#   def forward(self, x, edges):

#     x = self.linear1(x)
#     x = self.batch1(x)
#     x = F.elu(x)
#     x = self.linear2(x)
#     x = self.batch2(x)
#     x = F.elu(x)
#     x = self.linear3(x)
#     x = self.batch3(x)
#     x = F.elu(x)

#     x = self.GAT1(x,edges)
#     x = self.batch4(x)
#     x = F.elu(x)

#     for i, layer in enumerate(self.GAT_l):
#       x = layer(x, edges) #+ x
#       x = self.batch_l[i](x)
#       x = F.elu(x)
      
      
    
#     return x

class GATSY(nn.Module):
    """
    Graph Attention Network (GATSY) for artist similarity task.
    - Customizable GAT layers: number of GAT layers is controlled by `n_layers`.
    - Fixed back-end: two linear layers with batch normalization and ELU activations.
    - Batch normalization included after each layer for faster training and better convergence.
    """

    def __init__(self, n_heads, n_layers, input_dim=2613, hidden_dim=256, output_dim=256):
        """
        Args:
            n_heads (int): Number of attention heads in GAT layers.
            n_layers (int): Number of GAT layers.
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden feature dimensionality.
            output_dim (int): Output feature dimensionality.
        """
        super(GATSY, self).__init__()

        # GAT layers
        self.gat_layers = nn.ModuleList()
        #self.batch_norms = nn.ModuleList()

        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=n_heads, concat=True))
        #self.batch_norms.append(nn.BatchNorm1d(n_heads * hidden_dim))

        for _ in range(n_layers - 1):
            self.gat_layers.append(GATConv(n_heads * hidden_dim, hidden_dim, heads=n_heads, concat=True))
            #self.batch_norms.append(nn.BatchNorm1d(n_heads * hidden_dim))

        # Back-end: Two fixed linear layers
        self.linear1 = nn.Linear(n_heads * hidden_dim, hidden_dim)
        #self.batch1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edges):
        """
        Args:
            x (Tensor): Input node features of shape [num_nodes, input_dim].
            edges (Tensor): Edge indices of shape [2, num_edges].
        Returns:
            Tensor: Node embeddings of shape [num_nodes, output_dim].
        """
        # GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = F.elu(gat_layer(x, edges))
            x = F.normalize(x, p=2, dim=1)

        # Back-end linear layers
        x = F.elu(self.linear1(x))
        x = self.linear2(x)

        return x

class GraphSage(nn.Module):
  ''' This class is used for all the architectures described in the research, all the details are described in the paper '''
  def __init__(self, aggr = 'mean'):
    super(GraphSage, self).__init__()
    self.SG1 = SAGEConv(2613, 256, aggr = aggr, normalize = True, bias = True, project = True)
    self.SG2 = SAGEConv(256, 256, aggr = aggr, normalize = True, bias = True, project = True)
    self.SG3 = SAGEConv(256, 256, aggr = aggr, normalize = True, bias = True, project = True)

    self.FC1 = nn.Linear(256,256)
    self.FC2 = nn.Linear(256,256)
    self.FC3 = nn.Linear(256,100)
    
  
  def forward(self, x, edges):

    x = F.elu(self.SG1(x,edges))
    x = F.elu(self.SG2(x,edges))
    x = F.elu(self.SG3(x,edges))

    x = F.elu(self.FC1(x))
    x = F.elu(self.FC2(x))
    x = self.FC3(x)

    return x

class GINSY(nn.Module):
    """
    Graph Isomorphism Network (GINSY) for artist similarity task.
    - Customizable GIN layers: number of GIN layers is controlled by `n_layers`.
    - Fixed back-end: two linear layers with batch normalization and ELU activations.
    - Batch normalization included after each layer for faster training and better convergence.
    """

    def __init__(self, n_layers, input_dim=2613, hidden_dim=256, output_dim=256):
        """
        Args:
            n_layers (int): Number of GIN layers.
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden feature dimensionality.
            output_dim (int): Output feature dimensionality.
        """
        super(GINSY, self).__init__()

        # GIN layers
        self.gin_layers = nn.ModuleList()

        # First GIN layer
        self.gin_layers.append(
            GINConv(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        )

        # Subsequent GIN layers
        for _ in range(n_layers - 1):
            self.gin_layers.append(
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
            )

        # Back-end: Two fixed linear layers
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edges):
        """
        Args:
            x (Tensor): Input node features of shape [num_nodes, input_dim].
            edges (Tensor): Edge indices of shape [2, num_edges].
        Returns:
            Tensor: Node embeddings of shape [num_nodes, output_dim].
        """
        # GIN layers
        for gin_layer in self.gin_layers:
            x = F.relu(gin_layer(x, edges))
            x = F.normalize(x, p=2, dim=1)

        # Back-end linear layers
        x = F.elu(self.linear1(x))
        x = self.linear2(x)

        return x

class GCNSY(nn.Module):
    """
    Graph Convolutional Network (GCNSY) for artist similarity task.
    - Customizable GCN layers: number of GCN layers is controlled by `n_layers`.
    - Fixed back-end: two linear layers with batch normalization and ELU activations.
    - Batch normalization included after each layer for faster training and better convergence.
    """

    def __init__(self, n_layers, input_dim=2613, hidden_dim=256, output_dim=256):
        """
        Args:
            n_layers (int): Number of GCN layers.
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden feature dimensionality.
            output_dim (int): Output feature dimensionality.
        """
        super(GCNSY, self).__init__()

        # GCN layers
        self.gcn_layers = nn.ModuleList()

        # First GCN layer
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))

        # Subsequent GCN layers
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Back-end: Two fixed linear layers
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edges):
        """
        Args:
            x (Tensor): Input node features of shape [num_nodes, input_dim].
            edges (Tensor): Edge indices of shape [2, num_edges].
        Returns:
            Tensor: Node embeddings of shape [num_nodes, output_dim].
        """
        # GCN layers
        for gcn_layer in self.gcn_layers:
            x = F.relu(gcn_layer(x, edges))
            x = F.normalize(x, p=2, dim=1)

        # Back-end linear layers
        x = F.elu(self.linear1(x))
        x = self.linear2(x)

        return x

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree, homophily, to_undirected, is_undirected
from torch_geometric.utils import negative_sampling

from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from typing import Optional

# This code is partially inspired from the GRAFF implementation available at https://github.com/realfolkcode/GRAFF

class Symmetric(torch.nn.Module):
    def forward(self, w):
        # This class implements the method to define the symmetry in the squared matrices.
        return w.triu(0) + w.triu(1).transpose(-1, -2)
    
class PairwiseParametrization(torch.nn.Module):
    def forward(self, W):
        # Construct a symmetric matrix with zero diagonal
        # The weights are initialized to be non-squared, with 2 additional columns. We cut from two of these
        # two vectors q and r, and then we compute w_diag as described in the paper.
        # This procedure is done in order to easily distribute the mass in its spectrum through the values of q and r
        W0 = W[:, :-2].triu(1)

        W0 = W0 + W0.T

        # Retrieve the `q` and `r` vectors from the last two columns
        q = W[:, -2]
        r = W[:, -1]
        # Construct the main diagonal
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r)

        return W0 + w_diag

class External_W(nn.Module):
    def __init__(self, input_dim, device = 'cpu'):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((1, input_dim)))
        self.reset_parameters()
        self.to(device)
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w)

    def forward(self, x):
        # x * self.w behave like a diagonal matrix op., we multiply each row of x by the element-wise w
        return x * self.w


class Source_b(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.empty(1))
     
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        torch.nn.init.normal_(self.beta)
    


    def forward(self, x):
        return x * self.beta


class PairwiseInteraction_w(nn.Module):
    def __init__(self, input_dim, symmetry_type='1', device = 'cpu'):
        super().__init__()
        self.W = torch.nn.Linear(input_dim + 2, input_dim, bias = False)

        if symmetry_type == '1':
            symmetry = PairwiseParametrization()
        elif symmetry_type == '2':
            symmetry = Symmetric()

        parametrize.register_parametrization(
            self.W, 'weight', symmetry, unsafe=True)
        self.reset_parameters()
        self.to(device)
        
    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x):
        return self.W(x)

        

class GRAFFConv(MessagePassing):
    def __init__(self, external_w, source_b, pairwise_w, self_loops=True):
        super().__init__(aggr='add')

        self.self_loops = self_loops
        self.external_w = external_w #External_W(self.in_dim, device=device)
        self.beta = source_b #Source_b(device = device)
        self.pairwise_W = pairwise_w #PairwiseInteraction_w(self.in_dim, symmetry_type=symmetry_type, device=device)
   

    def forward(self, x, edge_index, x0):

        # We set the source term, which corrensponds with the initial conditions of our system.

        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        out_p = self.pairwise_W(x)

        out = self.propagate(edge_index, x=out_p)

        out = out - self.external_w(x) - self.beta(x0)

        return out

    def message(self, x_j, edge_index, x):
        # Does we need the degree of the row or from the columns?
        # x_i are the columns indices, whereas x_j are the row indices
        row, col = edge_index

        # Degree is specified by the row (outgoing edges)
        deg_matrix = degree(col, num_nodes=x.shape[0], dtype=x.dtype)
        deg_inv = deg_matrix.pow(-0.5)
        
        deg_inv[deg_inv == float('inf')] = 0

        denom_degree = deg_inv[row]*deg_inv[col]

        # Each row of denom_degree multiplies (element-wise) the rows of x_j
        return denom_degree.unsqueeze(-1) * x_j




class GRAFF(nn.Module):
    def __init__(self, num_layers, input_feat = 2613, hidden_dim = 256, normalize = True, step=0.1, symmetry_type='1', self_loops=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.enc = torch.nn.Linear(
            input_feat, hidden_dim, bias=False)

        self.external_w = External_W(hidden_dim, device=device)
        self.source_b = Source_b(device=device)
        self.pairwise_w = PairwiseInteraction_w(
            hidden_dim, symmetry_type=symmetry_type, device=device)

        self.GRAFF = GRAFFConv(self.external_w, self.source_b, self.pairwise_w,
                                 self_loops=self_loops)
        
        self.num_layers = num_layers
        
        self.normalize = normalize
        if self.normalize:
            self.batch1 = torch.nn.BatchNorm1d(hidden_dim)

        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)


        self.step = step



        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.external_w.reset_parameters()
        self.source_b.reset_parameters()
        self.pairwise_w.reset_parameters()


    def forward(self, x, edges):

        
        
        x = self.enc(x)

        if self.normalize:
            x = self.batch1(x)



        x0 = x.clone()
        
        
        for i in range(self.num_layers):

            x = x + self.step*F.elu(self.GRAFF(x, edges, x0))
            #x = x + self.step*(self.GRAFF(x, edge_index, x0))
        x = F.elu(self.linear1(x))
        x = self.linear2(x)    

        return x
    
