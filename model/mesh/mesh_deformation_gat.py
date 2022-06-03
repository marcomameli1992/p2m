import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv


class MeshDeformationBlock(nn.Module):
    '''
        Implementation of the mesh deformation block
    '''

    def __init__(self, transformer_name: str, feature_shape_dim) -> None:
        super(MeshDeformationBlock, self).__init__()

        feature_dict: dict = {
            'facebook/deit-base-patch16-384': 2304,
            'facebook/deit-tiny-distilled-patch16-224': 576,
            'facebook/deit-base-patch16-224': 2304,
            'facebook/deit-tiny-patch16-224': 576,
            'facebook/deit-small-patch16-224': 1152,
            'google/vit-base-patch16-224-in21k': 2304,
            'google/vit-base-patch16-224': 2304,
            'google/vit-base-patch16-384': 2304,
            'google/vit-base-patch32-384': 2304,
            'google/vit-base-patch32-224-in21k': 2304,
            'google/vit-large-patch16-224-in21k': 3072,
            'google/vit-large-patch16-224': 3072,
            'google/vit-large-patch32-224-in21k': 3072,
            'google/vit-large-patch32-384': 3072,
            'google/vit-large-patch16-384': 3072,
            'google/vit-huge-patch14-224-in21k': 3840

        }

        self.conv1 = GATv2Conv(feature_dict[transformer_name] + feature_shape_dim, 1024, heads=1, concat=False)
        self.conv21 = GATv2Conv(1024, 512, heads=1, concat=False)
        self.conv22 = GATv2Conv(512, 256, heads=1, concat=False)
        self.conv23 = GATv2Conv(256, 128, heads=1, concat=False)

        self.conv2 = [self.conv21, self.conv22, self.conv23]

        self.conv3 = GATv2Conv(128, 3)

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

