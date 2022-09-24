import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, concat=True, negative_slope=0.2):
        super(GResBlock, self).__init__()
        self.conv1 = GATv2Conv(in_channels, out_channels, heads, dropout, concat, negative_slope)
        self.conv2 = GATv2Conv(out_channels, out_channels, heads, dropout, concat, negative_slope)

    def forward(self, input, edge_index):
        x = self.conv1(input, edge_index)
        x = self.conv2(x, edge_index)
        return (x + input) * 0.5


class GBottleneck(nn.Module):
    def __init__(self, block_num, in_channels, out_channels, heads=1, dropout=0.0, concat=True, negative_slope=0.2):
        super(GBottleneck, self).__init__()
        blocks = [GResBlock(in_channels, out_channels, heads, dropout, concat, negative_slope)]
        for _ in range(block_num - 1):
            blocks.append(GResBlock(out_channels, out_channels, heads, dropout, concat, negative_slope))
        self.blocks = nn.Sequential(*blocks)
        self.conv1 = GATv2Conv(in_channels, out_channels, heads, dropout, concat, negative_slope)
        self.conv2 = GATv2Conv(out_channels, out_channels, heads, dropout, concat, negative_slope)

    def forward(self, input, edge_index):
        x = self.conv1(input, edge_index)
        x_cat = self.blocks(x, edge_index)
        x_out = self.conv2(x_cat, edge_index)
        return x_out, x_cat


class MeshDeformationBlock(nn.Module):

    def __init__(self, transformer_name: str, feature_shape_dim, intermedia_count: int = 6, last: bool = False) -> None:
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
        self.conv1 = GATv2Conv(feature_dict[transformer_name] + feature_shape_dim, 1024, heads=3, concat=True)
        self.conv2 = GBottleneck(intermedia_count, 1024, 1024, heads=3, concat=True)
        if last:
            self.conv3 = GATv2Conv(1024, 3)


    def forward(self, input, edge_index):
        x = self.conv1(input, edge_index)
        x, x_cat = self.conv2(x, edge_index)
        if self.conv3 is not None:
            x_3d = self.conv3(x, edge_index)
            return x, x_cat, x_3d
        else:
            return x, x_cat