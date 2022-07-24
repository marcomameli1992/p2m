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
            'facebook/deit-base-patch16-384': 36,
            'facebook/deit-tiny-distilled-patch16-224': 9,
            'facebook/deit-base-patch16-224': 36,
            'facebook/deit-tiny-patch16-224': 9,
            'facebook/deit-small-patch16-224': 18,
            'google/vit-base-patch16-224-in21k': 36,
            'google/vit-base-patch16-224': 36,
            'google/vit-base-patch16-384': 36,
            'google/vit-base-patch32-384': 36,
            'google/vit-base-patch32-224-in21k': 36,
            'google/vit-large-patch16-224-in21k': 48,
            'google/vit-large-patch16-224': 48,
            'google/vit-large-patch32-224-in21k': 48,
            'google/vit-large-patch32-384': 48,
            'google/vit-large-patch16-384': 48,
            'google/vit-huge-patch14-224-in21k': 48,
            'microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft': 48,
            'microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft': 36,
            'microsoft/swinv2-large-patch4-window12-192-22k': 36,
            'microsoft/swinv2-small-patch4-window8-256': 36,
            'microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft': 36,
            'microsoft/swinv2-base-patch4-window16-256': 36,
            'microsoft/swinv2-base-patch4-window12-192-22k': 36,
            'microsoft/swinv2-base-patch4-window8-256': 36,
            'microsoft/swinv2-small-patch4-window16-256': 36,
            'microsoft/swinv2-tiny-patch4-window16-256': 36,
            'microsoft/swinv2-tiny-patch4-window8-256': 39
            ,
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

