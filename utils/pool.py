import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


class FeaturePooling():

    def __init__(self, im) -> None:
        self.im = im

    def __call__(self, points, feat_extr, transformer: bool):
        # Project
        x, y = self._project(points)
        # Compute interpolated features
        feat_conv3, feat_conv4, feat_conv5 = feat_extr(self.im)

        interp_feat = self._pool_features(x, y, [feat_conv3, feat_conv4, feat_conv5], transformer)
        return interp_feat

    def _project(self, points):
        '''
            Project the 3D points onto the image plane using camera intrinsics
        '''

        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        focal = 248
        x = focal * torch.div(-Y, -Z) + 111.5
        y = focal * torch.div(X, -Z) + 111.5

        x = torch.clamp(x, 0, 223)
        y = torch.clamp(y, 0, 223)

        return x, y

    def _pool_features(self, x, y, feat_list, transformer: bool):
        '''
            Pool the features from four nearby pixels using bilinear interpolation
        '''
        concat_features = torch.FloatTensor().to(device)

        if not transformer:
            concat_features = torch.cat(feat_list, dim=2)
            if x.shape[0] > concat_features.shape[1]:
                concat_features = torch.cat((concat_features, ) * (int(round(x.shape[0]/concat_features.shape[1])) + 1), dim=1)
            return concat_features[0, 0:x.shape[0], :]
        else:
            for feat in feat_list:
                #feat = feat.t()
                if len(feat.shape) < 4:
                    feat = torch.unsqueeze(feat, dim=0)
                d = feat.shape[2] - 1  # range from 0 to d - 1

                x_ext = (x / 224) * d
                y_ext = (y / 224) * d
                x1 = torch.floor(x_ext).long()
                x2 = torch.ceil(x_ext).long()
                y1 = torch.floor(y_ext).long()
                y2 = torch.ceil(y_ext).long()

                # Pool the four nearby pixels features
                f_Q11 = feat[0, :, x1, y1]
                f_Q12 = feat[0, :, x1, y2]
                f_Q21 = feat[0, :, x2, y1]
                f_Q22 = feat[0, :, x2, y2]

                # Bilinear interpolation
                w1 = x2.float() - x_ext
                w2 = x_ext - x1.float()
                w3 = y2.float() - y_ext
                w4 = y_ext - y1.float()
                feat_bilinear = f_Q11 * w1 * w3 + f_Q21 * w2 * w3 + f_Q12 * w1 * w4 + f_Q22 * w2 * w4

                concat_features = torch.cat((concat_features, feat_bilinear))

            print(concat_features.shape)

            return concat_features.t()


if __name__ == "__main__":
    from graph import Graph
    import matplotlib.pyplot as plt

    print("Testing Feature Pooling")
    graph = Graph("./ellipsoid/init_info.pickle")
    pool = FeaturePooling(None)
    x, y = pool._project(graph.vertices)
    x = x.numpy()
    y = y.numpy()
    img = np.zeros((224, 224, 3), np.uint8)
    img[np.round(x).astype(int), np.round(y).astype(int), 2] = 0
    img[np.round(x).astype(int), np.round(y).astype(int), 1] = 255
    plt.imshow(img)
    plt.show()