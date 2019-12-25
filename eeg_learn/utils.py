import scipy.io
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import qhull

import tensorflow as tf
from tensorflow.keras.constraints import Constraint


def cartesian_to_spherical(x, y, z):
    x2_y2 = x ** 2 + y ** 2
    radius = math.sqrt(x2_y2 + z ** 2)
    elevation = math.atan2(z, math.sqrt(x2_y2))
    azimuth = math.atan2(y, x)
    return radius, elevation, azimuth


def polar_to_cartesian(theta, rho):
    return rho * math.cos(theta), rho * math.sin(theta)


def azimuthal_projection(position):
    [radius, elevation, azimuth] = cartesian_to_spherical(position[0], position[1], position[2])
    return polar_to_cartesian(azimuth, math.pi / 2 - elevation)


def reformat_input(data, labels, indices):
    np.random.shuffle(indices[0])
    np.random.shuffle(indices[0])
    train_indices = indices[0][len(indices[1]):]
    valid_indices = indices[0][:len(indices[1])]
    test_indices = indices[1]

    return [(data[train_indices], np.squeeze(labels[train_indices]).astype(np.int32)),
            (data[valid_indices], np.squeeze(labels[valid_indices]).astype(np.int32)),
            (data[test_indices], np.squeeze(labels[test_indices]).astype(np.int32))]


def load_data(data_folder):
    # Load electrode locations
    locations = scipy.io.loadmat(data_folder + '/Neuroscan_locs_orig.mat')
    locations_3d = locations['A']
    locations_2d = []
    # Convert to 2D
    for position in locations_3d:
        locations_2d.append(azimuthal_projection(position))

    features = scipy.io.loadmat(data_folder + '/FeatureMat_timeWin.mat')['features']
    subject_numbers = np.squeeze(scipy.io.loadmat(data_folder + '/trials_subNums.mat')['subjectNum'])

    # Leave-Subject-Out cross validation
    # 13 arrays each of which contain 2 arrays: 1 holds indices of 1 subject and the other array holds the rest indices
    fold_pairs = []
    for i in np.unique(subject_numbers):
        ts = subject_numbers == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr)
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))

    return features, subject_numbers, fold_pairs, np.array(locations_2d)


def compute_interpolation_weights(electrode_locations, grid):
    triangulation = qhull.Delaunay(electrode_locations)
    simplex = triangulation.find_simplex(grid)
    vertices = np.take(triangulation.simplices, simplex, axis=0)
    temp = np.take(triangulation.transform, simplex, axis=0)
    delta = grid - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, interpolation_weights, fill_value=np.nan):
    vtx, wts = interpolation_weights
    output = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    output[np.any(wts < 0, axis=1)] = fill_value
    return output


def convert_to_images(features, n_gridpoints, n_channels, channel_size, interpolation_weights):
    n_samples = features.shape[0]

    interpolated_data = np.zeros([n_channels, n_samples, n_gridpoints * n_gridpoints])

    for channel in range(n_channels):
        frequency_features = features[:, channel * channel_size: (channel + 1) * channel_size]
        for sample in range(n_samples):
            interpolated_data[channel, sample, :] = interpolate(frequency_features[sample], interpolation_weights)
    return interpolated_data.reshape((n_channels, n_samples, n_gridpoints, n_gridpoints))


class ImagePreprocessor:
    """
    Per channel scaling + nan imputing. StandardScaler ignores nan values here, so they don't impact mean and std
    """

    def __init__(self):
        self.is_fit = False

    def fit_transform(self, images):
        self.scalers = []
        for i in range(images.shape[-1]):
            scaler = StandardScaler()
            images[..., i] = self._scale_data(images, i, scaler)
            self.scalers.append(scaler)
        self.is_fit = True
        return np.nan_to_num(images)

    def transform(self, images):
        assert self.is_fit
        for i in range(images.shape[-1]):
            images[..., i] = self._scale_data(images, i, self.scalers[i])
        return np.nan_to_num(images)

    def _scale_data(self, images, i, scaler):
        channel_data = images[..., i].reshape(images.shape[0], -1)
        channel_data = scaler.fit_transform(channel_data)
        return channel_data.reshape(images.shape[:-1])


class WeightClip(Constraint):
    '''
    Clips the weights by value c
    '''

    def __init__(self, c):
        self.c = c

    def __call__(self, p):
        return tf.keras.backend.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}
