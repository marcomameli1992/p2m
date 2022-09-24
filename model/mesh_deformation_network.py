import torch.nn as nn
import torch
from model.mesh.GResBlock import MeshDeformationBlock
from model.image.transformer import Extractor
import numpy as np


class GraphNetwork(nn.Module):
    '''
        Implement the full cascaded mesh deformation network
    '''

    def __init__(self, transformer_name: str = 'google/vit-huge-patch14-224-in21k') -> None:
        super(GraphNetwork, self).__init__()

        self.feat_extr = Extractor(transformer_name)
        self.transf = True
        self.layer1 = MeshDeformationBlock(transformer_name, 3, last=True)
        self.layer2 = MeshDeformationBlock(transformer_name, 128, last=True)
        self.layer3 = MeshDeformationBlock(transformer_name, 128, last=True)

    def forward(self, graph, pool):
        # Initial ellipsoid mesh
        elli_points = graph.vertices.clone()

        # Layer 1
        features = pool(elli_points, self.feat_extr, self.transf)
        input = torch.cat((features, elli_points), dim=1)
        x1, x_cat1, coord1 = self.layer1(input, graph.adjacency_mat[0])
        graph.vertices = coord1

        # Unpool graph
        x = graph.unpool(x1)
        coord_1_1 = x.vertices.clone()

        # Layer 2
        features = pool(x.vertices, self.feat_extr, self.transf)
        input = torch.cat((features, x_cat1), dim=1)
        x2, x_cat2, coord2 = self.layer2(input, graph.adjacency_mat[1])
        graph.vertices = coord2

        # Unpool graph
        x = graph.unpool(x2)
        coord_2_1 = x.vertices.clone()

        # Layer 3
        features = pool(x.vertices, self.feat_extr, self.transf)
        input = torch.cat((features, x), dim=1)
        x3, x_cat3, coord3 = self.layer3(input, graph.adjacency_mat[2])
        graph.vertices = coord3

        return elli_points, coord1, coord_1_1, coord2, coord_2_1, coord3

    def get_nb_trainable_params(self):
        '''
            Return the number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])
