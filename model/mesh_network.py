import numpy as np
import torch
import torch.nn as nn

from model.mesh.mesh_deformation_gat import MeshDeformationBlock
from model.image.vgg import VGG
from model.image.transformer import Extractor


class GraphNetwork(nn.Module):
    '''
        Implement the full cascaded mesh deformation network
    '''



    def __init__(self, transformer_name: str = 'google/vit-huge-patch14-224-in21k') -> None:
        super(GraphNetwork, self).__init__()

        self.feat_extr = Extractor(transformer_name)
        self.transf = True
        self.layer1 = MeshDeformationBlock(transformer_name, 3)
        self.layer2 = MeshDeformationBlock(transformer_name, 128)
        self.layer3 = MeshDeformationBlock(transformer_name, 128)

    def forward(self, graph, pool):
        # Initial ellipsoid mesh
        elli_points = graph.vertices.clone()

        # Layer 1
        features = pool(elli_points, self.feat_extr, self.transf)
        input = torch.cat((features, elli_points), dim=1)
        x, coord_1 = self.layer1(input, graph.adjacency_mat[0])
        graph.vertices = coord_1

        # Unpool graph
        x = graph.unpool(x)
        coord_1_1 = graph.vertices.clone()

        # Layer 2
        features = pool(graph.vertices, self.feat_extr, self.transf)
        input = torch.cat((features, x), dim=1)
        x, coord_2 = self.layer2(input, graph.adjacency_mat[1])
        graph.vertices = coord_2

        # Unpool graph
        x = graph.unpool(x)
        coord_2_1 = graph.vertices.clone()

        # Layer 3
        features = pool(graph.vertices, self.feat_extr, self.transf)
        input = torch.cat((features, x), dim=1)
        x, coord_3 = self.layer3(input, graph.adjacency_mat[2])
        graph.vertices = coord_3

        return elli_points, coord_1, coord_1_1, coord_2, coord_2_1, coord_3

    def get_nb_trainable_params(self):
        '''
            Return the number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])