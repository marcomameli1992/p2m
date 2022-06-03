import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv


class MeshDeformationBlock(nn.Module):
    '''
        Implementation of the mesh deformation block
    '''

    def __init__(self, feature_shape_dim) -> None:
        super(MeshDeformationBlock, self).__init__()

        self.conv1 = GCNConv(1280 + feature_shape_dim, 1024)
        self.conv21 = GCNConv(1024, 512)
        self.conv22 = GCNConv(512, 256)
        self.conv23 = GCNConv(256, 128)

        self.conv2 = [self.conv21, self.conv22, self.conv23]

        self.conv3 = GCNConv(128, 3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        '''
            Return 3D shape features (return[0]) and predicted 3D coordinates
            of the vertices (return[1])
        '''

        out = self.conv1(x, edge_index)
        out = self.relu(out)

        for i in range(len(self.conv2)):
            conv = self.conv2[i]
            out = conv(out, edge_index)
            out = self.relu(out)

        return out, self.conv3(out, edge_index)

