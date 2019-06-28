from threading import Thread
from threading import Lock

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
from torch.autograd import Variable

import scipy.ndimage as nd
import scipy.io as io
from scipy.stats import multivariate_normal
from scipy.ndimage.filters import convolve
from collections import OrderedDict

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os
import pickle
import wandb
import glob
from io import StringIO
from scipy.ndimage import rotate

from common.torch_utils import SubsetIterativeSampler, EuclideanDistanceLoss
from common.torch_utils import FocalLoss, BCELossClassWeighted
from common.torch_utils import DiceLossSampleWeighted, DiceLoss, DiceLossPerSample, DiceLossVoxelWeighted
from common.torch_utils import DiceLossSampleVoxelWeighted

LEFT_RIGHT_LANDMARK_MAPPING29 = [15, 18, 17, 16, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                                 4, 00, 3, 2, 1, 25, 24, 23, 22, 21, 20, 19, 28, 27, 26]
LEFT_RIGHT_LANDMARK_MAPPING28 = [15, 18, 17, 16, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 0, 3, 2,
                                 1, 25, 24, 23, 22, 21, 20, 19, 27, 26]


def save_voxel_plot(voxels, path, args, iteration='test', elevation=50, azimith=90, postfix='',
                    titles=None, landmarks=None, mode='train', rows=2):
    os.makedirs(path, exist_ok=True)

    voxels = voxels.__ge__(0.5)
    num = max(1, voxels.shape[0] // rows)

    if args.use_matplotlib:
        png_file = path + '/{}{}.png'.format(str(iteration).zfill(3), postfix)
        fig = plt.figure(figsize=(32, 16))
        gs = gridspec.GridSpec(rows, num)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(voxels):
            x, y, z = sample.nonzero()
            ax = plt.subplot(gs[i], projection='3d')
            ax.scatter(x, y, z, zdir='z', c='red')
            if landmarks is not None:
                landmark = landmarks[i]
                ax.scatter(landmark[:, 0], landmark[:, 1], landmark[:, 2], c='blue')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xlim(0, args.cube_len)
            ax.set_ylim(0, args.cube_len)
            ax.set_zlim(0, args.cube_len)
            ax.set_aspect('equal')
            ax.view_init(elevation, azimith)

            if titles is not None:
                if i < len(titles):
                    ax.set_title(titles[i])
        plt.savefig(png_file, bbox_inches='tight')
        plt.close()

        if args.use_wandb:
            from PIL import Image
            fig.canvas.draw()
            ncols, nrows = fig.canvas.get_width_height()
            im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
            im = Image.fromarray(im)
            wandb_image = wandb.Image(im, caption=str(iteration))
            wandb.log({'Examples_' + mode: wandb_image})

    if args.use_plotly and \
            titles is not None:
        # assuming the next n/2 of samples are generated, and the first n/2 are ground truth
        for i in range(num):
            if landmarks:
                save_plotly_html(voxels[i], voxels[i + num],
                                 path + '/{}{}_{}.html'.format(
                                     str(iteration).zfill(3), postfix, i), titles[i],
                                 landmarks[i], landmarks[i + num])
            else:
                save_plotly_html(voxels[i], voxels[i + num],
                                 path + '/{}{}_{}.html'.format(
                                     str(iteration).zfill(3), postfix, i), titles[i])

    if args.save_voxels:
        with open(path + '/{}{}.pkl'.format(str(iteration).zfill(3), postfix), "wb") as f:
            pickle.dump((voxels, landmarks), f, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_visualization(data_list):
    '''
    Prepares data for visualization by creating a list of lists.
    :param data_list: A list of numpy arrays, all with the same first dimension.
    :return: A list of lists, where each item holds information about a sinlg element.
    '''
    result = []
    for j in range(data_list[0].shape[0]):
        result.append([])
        for i in range(len(data_list)):
            result[-1].append(data_list[i][j])

    return result


def save_voxel_plot2(data_list, path, args, iteration='test', elevation=50, azimith=90, postfix='',
                     titles=None, mode='train'):
    os.makedirs(path, exist_ok=True)
    rows = len(data_list[0])
    num = len(data_list)

    if args.use_matplotlib:
        png_file = path + '/{}{}.png'.format(str(iteration).zfill(3), postfix)
        fig = plt.figure(figsize=(32, 16))
        gs = gridspec.GridSpec(rows, num)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(data_list):
            for j, sub_sample in enumerate(sample):
                sub_sample = sub_sample.__ge__(0.5)
                x, y, z = sub_sample.nonzero()
                ax = plt.subplot(gs[j * num + i], projection='3d')
                ax.scatter(x, y, z, zdir='z', c='red')

                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xlim(0, args.cube_len)
                ax.set_ylim(0, args.cube_len)
                ax.set_zlim(0, args.cube_len)
                ax.set_aspect('equal')
                ax.view_init(elevation, azimith)

                if titles is not None:
                    ax.set_title(titles[i])

                if args.save_voxels:
                    pkl_file = os.path.join(path, '{}_{}_{}.npy'.format(
                        str(iteration).zfill(3), i, j))
                    # with open(pkl_file, "wb") as f:
                    #     pickle.dump(sub_sample, f, protocol=pickle.HIGHEST_PROTOCOL)
                    np.save(pkl_file, sub_sample)

            if args.use_plotly and titles is not None:
                save_plotly_html(sample[0], sample[1],
                                 path + '/{}{}_{}.html'.format(
                                     str(iteration).zfill(3), postfix, i), titles[i])

        plt.savefig(png_file, bbox_inches='tight')
        plt.close()

        if args.use_wandb:
            from PIL import Image
            fig.canvas.draw()
            ncols, nrows = fig.canvas.get_width_height()
            im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
            im = Image.fromarray(im)
            wandb_image = wandb.Image(im, caption=str(iteration))
            wandb.log({'Examples_' + mode: wandb_image})


def save_plotly_html(sample, sample2, filename, title, landmark=None, landmark2=None):
    # todo: update to include varargs and plot all in the same plot
    import plotly.plotly as py
    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

    point_size = 5
    data = []

    if landmark:
        labels = [str(i) for i in range(1, landmark.shape[0] + 1)]
        trace2 = go.Scatter3d(
            x=landmark[:, 0],
            y=landmark[:, 1],
            z=landmark[:, 2],
            mode='text',
            text=labels,
            marker=dict(
                size=20,
                line=dict(
                    color='rgba(255, 0, 217, 1)',
                    width=20
                ),
                opacity=1
            ),
            name='landmarks_names'
        )
        data.append(trace2)

        trace3 = go.Scatter3d(
            x=landmark[:, 0],
            y=landmark[:, 1],
            z=landmark[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                line=dict(
                    color='rgba(255, 0, 217, 1)',
                    width=0
                ),
                opacity=1
            ),
            name='landmarks1'
        )
        data.append(trace3)

    if landmark2:
        trace5 = go.Scatter3d(
            x=landmark2[:, 0],
            y=landmark2[:, 1],
            z=landmark2[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                line=dict(
                    color='rgba(100, 200, 217, 1)',
                    width=0
                ),
                opacity=1
            ),
            name='landmarks2'
        )
        data.append(trace5)

    x, y, z = sample.nonzero()
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=point_size,
            line=dict(
                color='rgba(0, 255, 0, 1)',
                width=0
            ),
            opacity=0.1
        ),
        name='sample1'

    )
    data.append(trace1)

    x, y, z = sample2.nonzero()
    trace4 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=point_size,
            line=dict(
                color='rgba(255, 0, 0, 1)',
                width=0
            ),
            opacity=0.1
        ),
        name='sample2'
    )
    data.append(trace4)

    layout = go.Layout(
        title=title,
        titlefont=dict(size=40),
        font=dict(family='Courier New, monospace', size=18, color='#7f557f'),
    )
    plot(go.Figure(data=data, layout=layout), filename=filename, auto_open=False)


def make_hyparam_string_gen(args):
    hyparam_list = [("model", args.model_name),
                    ("alg", args.alg_name),
                    ("postfix", args.postfix)]
    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))

    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]


class PickleDataset(data.Dataset):
    """Custom dataset designed to load processed data saved as pickle files."""

    def __init__(self, data_path):
        if not os.path.isdir(data_path):
            raise NotImplementedError('Path to the pickle dataset is expected to be a directory '
                                      'of pickle files, each containing a processed sample.')
        self.data, self.keys = PickleDataset.load_pkls(data_path)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['input'].astype(np.float32), \
               sample['target'].astype(np.float32), \
               sample['box'].astype(np.float32)

    @staticmethod
    def load_pkls(path):
        files = sorted(glob.glob(os.path.join(path, '*.pkl')))
        keys = pickle.load(open(files[0], 'rb')).keys()
        print('Pickle dataset keys:', keys)
        data_arr = []
        for file in files:
            print(file)
            data_arr.append(pickle.load(open(file, 'rb')))
        return data_arr, keys

    def __len__(self):
        return len(self.data)


class NumpyDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, data_path, labels_path, args, augment=True):
        """Loads a numpy dataset
        Args:
            data_path: Path to numpy dataset
            labels_path: Path to labels of the dataset, if any.
            args: Dictionary holding hyperparameters including resolution
            which defines how to downsample the dataset.
        """
        self.args = args
        self.augment = augment

        try:
            if os.path.isdir(labels_path):
                self.labels = NumpyDataset.load_nps(labels_path)
            else:
                self.labels = np.load(labels_path) if labels_path else None
        except TypeError:
            self.labels = None

        if os.path.isdir(data_path):
            data_unprocessed = NumpyDataset.load_nps(data_path)
            self.data, self.labels = self.process_nps(data_unprocessed, self.labels, args.cube_len)
        else:
            self.data = np.load(data_path)

        res = int(args.data_resolution)
        if res > 1:
            data_down = np.zeros((self.data.shape[0],
                                  int(self.data.shape[1] // res),
                                  int(self.data.shape[2] // res),
                                  int(self.data.shape[3] // res)))
            k = np.ones((res, res, res)) / (res ** 3)
            for i in range(self.data.shape[0]):
                data_down[i] = convolve(self.data[i], k, cval=0, mode='constant')[0:-1:res, 0:-1:res, 0:-1:res]
            self.data = data_down

        print('[Numpy Data loaded] Data Shape:', self.data.shape,
              'Resolution:', args.data_resolution, 'mm')
        if self.labels is not None:
            assert self.labels.shape[0] == self.data.shape[0], 'Same number of data and labels is expected.'
            if res > 1:
                self.labels /= res
            print('Labels Shape:', self.labels.shape)

    def __getitem__(self, index):
        # print('index:', index)
        volume = self.data[index, :, :, :]
        assert volume.max() == 1, 'Volume is supposed to be binary with values 0 and 1 (occupancy map)'

        landmark = None
        if self.labels is not None:
            landmark = self.labels[index]

        return self.preprocess(volume, landmark=landmark)

    def preprocess(self, volume, landmark=None):
        if self.args.multi_target:
            num_targets = self.args.num_targets
            # todo: enforce others not to be from validation data and not the same as the current index
            others_idx = np.random.permutation(self.data.shape[0])[:num_targets]
            others = self.data[others_idx]

            # now include the current volume too
            others = np.copy(np.concatenate((others, [np.copy(volume)]), axis=0))
        else:
            others = np.array([])

        if self.augment:
            if self.args.augment_mirror:
                if np.random.rand() > 0.5:
                    volume, landmark = mirror_numpy(X=volume, Y=landmark, dim_size=self.args.cube_len, dim=0)
                    others, _ = mirror_numpy(X=others, dim_size=self.args.cube_len, dim=0)

            if self.args.augment_rotate:
                if np.random.rand() > 0.5:
                    degrees = tuple(np.random.randint(-self.args.augment_rotate_degree,
                                                      self.args.augment_rotate_degree,
                                                      3))
                    volume, landmark = rotate_numpy(X=volume, Y=landmark, degrees=degrees, dims=(2,))
                    for o in range(others.shape[0]):
                        others[o], _ = rotate_numpy(X=others[o], degrees=degrees, dims=(2,))

            if self.args.augment_trans:
                if np.random.rand() > 0.5:
                    num_voxels = tuple(np.random.randint(-self.args.augment_trans_voxels,
                                                         self.args.augment_trans_voxels,
                                                         1))
                    volume, landmark = shift_numpy(X=volume, Y=landmark, dim_size=self.args.cube_len,
                                                   num_voxels=num_voxels, dims=(0, 1, 2))
                    for o in range(others.shape[0]):
                        others[o], _ = shift_numpy(X=others[o], dim_size=self.args.cube_len,
                                                   num_voxels=num_voxels, dims=(0, 1, 2))

        if self.args.random_remove:
            volume_kept, volume_removed, box, gauss, others, others_weight = \
                random_remove(volume,
                              self.args.cube_len,
                              min_remove_width=self.args.remove_width_min,
                              max_remove_width=self.args.remove_width_max,
                              min_remove_voxels=self.args.min_remove_voxels,
                              min_rotate_deg=self.args.min_remove_rotate,
                              max_rotate_deg=self.args.max_remove_rotate,
                              weighted=(self.args.gen_g_loss[0] == 'weighted_dice' or
                                        self.args.gen_g_loss[0] == 'variational_weighted_dice'),
                              gauss_var=self.args.gaussian_variance,
                              others=others,
                              args=self.args)

            return torch.as_tensor(volume_kept, dtype=torch.float), \
                   torch.as_tensor(volume_removed, dtype=torch.float), \
                   torch.as_tensor(box, dtype=torch.float), \
                   torch.as_tensor(gauss, dtype=torch.float), \
                   torch.as_tensor(others, dtype=torch.float), \
                   torch.as_tensor(others_weight, dtype=torch.float)

        return torch.FloatTensor(volume) if landmark is None else \
            (torch.FloatTensor(volume), torch.FloatTensor(landmark))

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def load_nps(path):
        files = sorted(glob.glob(os.path.join(path, '*.npy')))
        # for i, file in enumerate(files):
        #     print(i, file)
        data_arr = []
        for file in files:
            data_arr.append(np.load(file).astype(dtype=np.float32))
        return data_arr

    def process_nps(self, data_list, label_list, cube_len):
        if label_list is not None:
            assert len(data_list) == len(label_list), '{} <> {}'.format(len(data_list), len(label_list))
            label_array = np.asarray(label_list)

        num = len(data_list)
        print('---Equalizing samples---')
        maxs = [cube_len, cube_len, cube_len]

        if not self.args.uniform_cube:
            data_array = np.zeros(([num] + maxs), dtype=np.float32)
        else:
            data_array = np.zeros(([num] + [max(maxs)] * 3), dtype=np.float32)

        for i, sample in enumerate(data_list):
            # Print warning if sample is bigger than cube_len
            if max(sample.shape) > cube_len:
                print('Sample #{} is bigger than cube_len.')

            # centralize mandible in the voxel space
            if not self.args.uniform_cube:
                start_index = [max(0, (maxs[d] - sample.shape[d])) // 2 for d in range(3)]
            else:
                start_index = [max(0, (max(maxs) - sample.shape[d])) // 2 for d in range(3)]

            data_array[i,
            start_index[0]: start_index[0] + sample.shape[0],
            start_index[1]: start_index[1] + sample.shape[1],
            start_index[2]: start_index[2] + sample.shape[2]] = sample[:cube_len, :cube_len, :cube_len]

            if label_list:
                label_array[i] += start_index
            else:
                label_array = None

        return data_array, label_array


def get_dataset_type(data_path):
    files = glob.glob(os.path.join(data_path, '*'))
    extensions = {}
    for file in files:
        ex = os.path.splitext(file)[1][1:]
        if ex in extensions.keys():
            extensions[ex] += 1
        else:
            extensions[ex] = 1

    max_ex = None
    max_ex_count = -1
    for key, val in extensions.items():
        if val > max_ex_count:
            max_ex_count = val
            max_ex = key

    print('max_ex', max_ex)
    if max_ex == 'npy':
        return 'numpy'
    elif max_ex == 'pkl':
        return 'pickle'
    else:
        raise NotImplementedError


def load_data(data_path, labels_path, args):
    print('[---- Loading Data ----]')
    print('Data Path:', data_path, '\nLabels Path:', labels_path)

    if not os.path.exists(data_path):
        raise FileExistsError('Data path ({}) not found.'.format(data_path))

    if args.dataset_type is None:
        args.dataset_type = get_dataset_type(data_path)

    if args.dataset_type == 'numpy':
        dsets = NumpyDataset(data_path, labels_path, args, augment=(not args.test))
    elif args.dataset_type == 'pickle':
        dsets = PickleDataset(data_path)
    else:
        raise NotImplementedError('Only numpy data format is supported.')

    data_size = dsets.__len__()

    # Split into train and valid
    if args.split_point == 'random':
        # if splitting into train-valid is random, override shuffle to False
        perm = np.random.permutation(data_size)
    elif args.split_point == 'iterative':
        # if splitting point is sequential, keep the shuffle settings
        perm = np.asarray(range(data_size))
    else:
        raise NotImplementedError

    num_train = int(args.split_ratio * data_size)
    if args.shuffle:
        train_sampler = SubsetRandomSampler(perm[:num_train])
    else:
        train_sampler = SubsetIterativeSampler(perm[:num_train])
    valid_sampler = SubsetIterativeSampler(perm[num_train:])
    print('Sample indices 1, size={}, {}:'.format(len(perm[:num_train]), perm[:num_train]))
    print('Sample indices 2, size={}: {}'.format(len(perm[num_train:]), perm[num_train:]))
    return dsets, train_sampler, valid_sampler


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_data_loaders(data_path, labels_path, args):
    dsets, train_sampler, valid_sampler = load_data(data_path, labels_path, args)
    dset_loaders1 = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, sampler=train_sampler,
                                                worker_init_fn=worker_init_fn)
    dset_loaders2 = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=False,
                                                num_workers=1, sampler=valid_sampler)
    return dset_loaders1, dset_loaders2


def loss_function(loss_str, wbce_weights=None, focal_gamma=None):
    if loss_str == 'bce':
        return torch.nn.BCELoss()
    elif loss_str == 'wbce':
        return BCELossClassWeighted(class_weight=wbce_weights)
    elif loss_str == 'dice':
        return DiceLoss()
    elif loss_str == 'variational_weighted_dice':
        return DiceLossSampleVoxelWeighted()
    elif loss_str == 'variational_dice':
        return DiceLossSampleWeighted()
    elif loss_str == 'dice_per_sample':
        return DiceLossPerSample()
    elif loss_str == 'weighted_dice':
        return DiceLossVoxelWeighted()
    elif loss_str == 'focal':
        return FocalLoss(focal_gamma)
    elif loss_str == 'l1':
        return torch.nn.L1Loss()
    elif loss_str == 'l2':
        return torch.nn.MSELoss()
    elif loss_str == 'euclidean':
        return EuclideanDistanceLoss()
    else:
        raise NotImplementedError


def mirror_torch(X, Y=None, dim_size=140, dim=0):
    '''
    :param X:
    :param Y:
    :param dim: The dimension (x=0, y=1, z=2) to flip the 3D data. This is not the same as axis in X
    :param dim_size: The number of voxels in dim
    :return:
    '''
    if len(X.size()) == 4:
        # first axis of X is the batch samples
        X = X.flip(dims=(dim + 1,))

        if Y is not None:
            if Y.size(1) == 28:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING28
            elif Y.size(1) == 29:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING29
            else:
                raise NotImplementedError('Only 28 and 29 landmarks are supported for mirroring.')

            Y[:, :, dim] = dim_size - Y[:, :, dim]
            Y = Y[:, LEFT_RIGHT_LANDMARK_MAPPING, :]
    elif len(X.size()) == 3:
        # single sample to mirror
        X = X.flip(dims=(dim,))

        if Y is not None:
            if Y.size(0) == 28:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING28
            elif Y.size(0) == 29:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING29
            else:
                raise NotImplementedError('Only 28 and 29 landmarks are supported for mirroring.')

            Y[:, dim] = dim_size - Y[:, dim]
            Y = Y[LEFT_RIGHT_LANDMARK_MAPPING, :]
    else:
        raise NotImplementedError

    return X, Y


def mirror_numpy(X, Y=None, dim_size=140, dim=0):
    '''
    :param X:
    :param Y:
    :param dim: The dimension (x=0, y=1, z=2) to flip the 3D data. This is not the same as axis in X
    :param dim_size: The number of voxels in dim
    :return:
    '''

    if len(X.shape) == 4:
        # first axis of X is the batch samples
        X = np.flip(X, axis=(dim + 1,))

        if Y is not None:
            if Y.shape[1] == 28:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING28
            elif Y.shape[1] == 29:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING29
            else:
                raise NotImplementedError('Only 28 and 29 landmarks are supported for mirroring.')

            Y[:, :, dim] = dim_size - Y[:, :, dim]
            Y = Y[:, LEFT_RIGHT_LANDMARK_MAPPING, :]
    elif len(X.shape) == 3:
        # single sample to mirror
        X = np.flip(X, axis=(dim,))

        if Y is not None:
            if Y.shape[0] == 28:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING28
            elif Y.shape[0] == 29:
                LEFT_RIGHT_LANDMARK_MAPPING = LEFT_RIGHT_LANDMARK_MAPPING29
            else:
                raise NotImplementedError('Only 28 and 29 landmarks are supported for mirroring.')

            Y[:, dim] = dim_size - Y[:, dim]
            Y = Y[LEFT_RIGHT_LANDMARK_MAPPING, :]
    elif len(X.shape) == 1:
        return X
    else:
        raise NotImplementedError

    X = np.ascontiguousarray(X)
    return X, Y


def shift(X, Y=None, dim_size=140, num_voxels=None, dims=None):
    '''
    :param X:
    :param Y:
    :param num_voxels: Single value or tuple
    :param dims: The dimension (x=0, y=1, z=2) to shift the 3D data. Single value or tuple
    :return:
    '''

    if len(X.size()) == 4:
        # first axis of X is the batch samples
        for iter_dim, vox in zip(dims, num_voxels):
            if iter_dim == 0:
                if vox > 0:
                    X[:, -vox:, :, :] = 0
                elif vox < 0:
                    X[:, :-vox, :, :] = 0
            elif iter_dim == 1:
                if vox > 0:
                    X[:, :, -vox:, :] = 0
                elif vox < 0:
                    X[:, :, :-vox, :] = 0
            elif iter_dim == 2:
                if vox > 0:
                    X[:, :, :, -vox:] = 0
                elif vox < 0:
                    X[:, :, :, :-vox] = 0

        dim_X = tuple([t + 1 for t in dims])
        X = torch.roll(X, num_voxels, dims=dim_X)

        if Y is not None:
            for iter_dim, vox in zip(dims, num_voxels):
                Y[:, :, iter_dim] += vox
            Y[Y < 0] = 0
            Y[Y > dim_size] = dim_size

    elif len(X.size()) == 3:
        # single sample to mirror
        for iter_dim, vox in zip(dims, num_voxels):
            if iter_dim == 0:
                if vox > 0:
                    X[-vox:, :, :] = 0
                elif vox < 0:
                    X[:-vox, :, :] = 0
            elif iter_dim == 1:
                if vox > 0:
                    X[:, -vox:, :] = 0
                elif vox < 0:
                    X[:, :-vox, :] = 0
            elif iter_dim == 2:
                if vox > 0:
                    X[:, :, -vox:] = 0
                elif vox < 0:
                    X[:, :, :-vox] = 0

        X = torch.roll(X, num_voxels, dims=dims)

        if Y is not None:
            for iter_dim, vox in zip(dims, num_voxels):
                Y[:, iter_dim] += vox
            Y[Y < 0] = 0
            Y[Y > dim_size] = dim_size
    elif len(X.shape) == 1:
        return X
    else:
        raise NotImplementedError

    return X, Y


def rotate_numpy(X, Y=None, degrees=None, dims=None):
    if Y is not None:
        # Have not yet implemented the rotation of landmark points
        from scipy.spatial.transform import Rotation as R
        r = R.from_rotvec([0, 0, degrees[0] / 180.0 * np.pi])
        # move landmarks to origin
        half = X.shape[-1] // 2
        Y -= half
        # rotate around origin
        Y = r.apply(Y)
        # move them back
        Y += half

    if len(X.shape) == 4:
        inc_dim = 1  # first axis of X is the batch samples
    elif len(X.shape) == 3:
        inc_dim = 0
    elif len(X.shape) == 1:
        return X
    else:
        raise NotImplementedError

    for iter_dim, deg in zip(dims, degrees):
        if iter_dim == 0:
            X = rotate(X, deg, axes=(1 + inc_dim, 2 + inc_dim), reshape=False, mode='constant',
                       cval=0, prefilter=False, order=2)
        elif iter_dim == 1:
            X = rotate(X, deg, axes=(0 + inc_dim, 2 + inc_dim), reshape=False, mode='constant',
                       cval=0, prefilter=False, order=2)
        elif iter_dim == 2:
            X = rotate(X, deg, axes=(0 + inc_dim, 1 + inc_dim), reshape=False, mode='constant',
                       cval=0, prefilter=False, order=2)

    return X, Y


def shift_numpy(X, Y=None, dim_size=140, num_voxels=None, dims=None):
    '''
    :param X:
    :param Y:
    :param num_voxels: Single value or tuple
    :param dims: The dimension (x=0, y=1, z=2) to shift the 3D data. Single value or tuple
    :return:
    '''

    if len(X.shape) == 4:
        # first axis of X is the batch samples
        for iter_dim, vox in zip(dims, num_voxels):
            if iter_dim == 0:
                if vox > 0:
                    X[:, -vox:, :, :] = 0
                elif vox < 0:
                    X[:, :-vox, :, :] = 0
            elif iter_dim == 1:
                if vox > 0:
                    X[:, :, -vox:, :] = 0
                elif vox < 0:
                    X[:, :, :-vox, :] = 0
            elif iter_dim == 2:
                if vox > 0:
                    X[:, :, :, -vox:] = 0
                elif vox < 0:
                    X[:, :, :, :-vox] = 0

        dim_X = tuple([t + 1 for t in dims])
        X = np.roll(X, num_voxels, axis=dim_X)

        if Y is not None:
            for iter_dim, vox in zip(dims, num_voxels):
                Y[:, :, iter_dim] += vox
            Y[Y < 0] = 0
            Y[Y > dim_size] = dim_size

    elif len(X.shape) == 3:
        # single sample to mirror
        for iter_dim, vox in zip(dims, num_voxels):
            if iter_dim == 0:
                if vox > 0:
                    X[-vox:, :, :] = 0
                elif vox < 0:
                    X[:-vox, :, :] = 0
            elif iter_dim == 1:
                if vox > 0:
                    X[:, -vox:, :] = 0
                elif vox < 0:
                    X[:, :-vox, :] = 0
            elif iter_dim == 2:
                if vox > 0:
                    X[:, :, -vox:] = 0
                elif vox < 0:
                    X[:, :, :-vox] = 0

        X = np.roll(X, num_voxels, axis=dims)

        if Y is not None:
            for iter_dim, vox in zip(dims, num_voxels):
                Y[:, iter_dim] += vox
            Y[Y < 0] = 0
            Y[Y > dim_size] = dim_size

    else:
        raise NotImplementedError

    return X, Y


# def shift_numpy(X, Y=None, dim_size=140, num_voxels=None, dims=None):
#     '''
#     :param X:
#     :param Y:
#     :param num_voxels: Single value or tuple
#     :param dim: The dimension (x=0, y=1, z=2) to shift the 3D data. Single value or tuple
#     :return:
#     '''
#     for iter_dim, vox in zip(dims, num_voxels):
#         if iter_dim == 0:
#             X[:, -vox:, :, :] = 0
#         elif iter_dim == 1:
#             X[:, :, -vox:, :] = 0
#         elif iter_dim == 2:
#             X[:, :, :, -vox:] = 0
#
#     dim_X = tuple([t + 1 for t in dims])
#     X = np.roll(X, num_voxels, axis=dim_X)
#
#     for iter_dim, vox in zip(dims, num_voxels):
#         Y[:, :, iter_dim] += vox
#
#     Y[Y < 0] = 0
#     Y[Y > dim_size] = dim_size
#
#     return X, Y


def var_or_cuda(x, device=0):
    if torch.cuda.is_available():
        x = x.cuda(device)
    return Variable(x)


def generateZ(args):
    if args.z_dis == "norm":
        Z = var_or_cuda(torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33))
    elif args.z_dis == "uni":
        Z = var_or_cuda(torch.randn(args.batch_size, args.z_size))
    else:
        print("z_dist is not normal or uniform")

    return Z


def dice(generated, real, threshold=0.5):
    '''
    Calculates dice between all samples of generated and real. Input could be grayscale,
    which will be binarized with the threhold value.

    :param generated: A batch of generated samples
    :param real: a batch of real samples
    :return: Dice score
    '''
    generated[generated >= threshold] = 1
    generated[generated < threshold] = 0
    real[real >= threshold] = 1
    real[real < threshold] = 0

    generated = generated.astype(np.int)
    real = real.astype(np.int)

    num_samples = generated.shape[0]
    results = np.zeros(num_samples)

    cross_mat = (generated & real)
    g_nonzero = generated.nonzero()[0]
    r_nonzero = real.nonzero()[0]

    for i in range(num_samples):
        tp = cross_mat[i].nonzero()[0].shape[0]
        size_gen = sum(g_nonzero == i)
        size_real = sum(r_nonzero == i)
        results[i] = (2 * tp) / (size_gen + size_real + 0.000001)

    return results


def dice_single_sample(generated, real):
    '''
    Calculates dice between single "binary" generated and real samples
    :param generated: A batch of generated samples
    :param real: a batch of real samples
    :return: Dice score
    '''
    generated = generated.astype(np.int)
    real = real.astype(np.int)

    cross_mat = (generated & real)

    tp = cross_mat.nonzero()[0].shape[0]
    size_gen = generated.nonzero()[0].shape[0]
    size_real = real.nonzero()[0].shape[0]
    results = (2 * tp) / (size_gen + size_real + 0.000001)

    return results


########################## Pickle helper ###############################


def read_pickles_dir(path, G=None, G_solver=None, D_=None, D_solver=None):
    try:
        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print('Iter:', recent_iter, 'Loaded from:', path)

        if G:
            with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
                G.load_state_dict(torch.load(f))
        if G_solver:
            with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
                G_solver.load_state_dict(torch.load(f))
        if D_:
            with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
                D_.load_state_dict(torch.load(f))
        if D_solver:
            with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
                D_solver.load_state_dict(torch.load(f))

    except FileNotFoundError as e:
        print('Trained model does not exist to load:', e)
        raise e
    except Exception as e:
        print("fail try read_pickle", e)
        raise e


def read_pickles_args(args, net1=None, solver1=None, net2=None, solver2=None):
    try:
        if net1 and args.load_model_path:
            with open(args.load_model_path, "rb") as f:
                net1.load_state_dict(torch.load(f))
            print('Loaded:', args.load_model_path)

        if solver1 and args.load_optim_path:
            with open(args.load_optim_path, "rb") as f:
                solver1.load_state_dict(torch.load(f))
            print('Loaded:', args.load_optim_path)

        if net2 and args.load_model_path2:
            with open(args.load_model_path2, "rb") as f:
                net2.load_state_dict(torch.load(f))
            print('Loaded:', args.load_model_path2)

        if solver2 and args.load_optim_path2:
            with open(args.load_optim_path2, "rb") as f:
                solver2.load_state_dict(torch.load(f))
            print('Loaded:', args.load_optim_path2)

    except FileNotFoundError as e:
        print('Trained model does not exist to load:', e)
        raise e
    except Exception as e:
        print("Fail read_pickle:", e)
        raise e


def read_pickle_path(path, net):
    try:
        with open(path, "rb") as f:
            net.load_state_dict(torch.load(f))
        print('Loaded:', path)
    except FileNotFoundError as e:
        print('Trained model does not exist to load:', e)
    except Exception as e:
        print("fail try read_pickle", e)


def save_new_pickle(path, iteration, G=None, G_solver=None, D_=None, D_solver=None, iter_append=True):
    if not os.path.exists(path):
        os.makedirs(path)

    if iter_append:
        append_str = str(iteration) + '.pkl'
    else:
        append_str = '0.pkl'

    if G:
        with open(path + "/G_" + append_str, "wb") as f:
            torch.save(G.state_dict(), f)
    if G_solver:
        with open(path + "/G_optim_" + append_str, "wb") as f:
            torch.save(G_solver.state_dict(), f)
    if D_:
        with open(path + "/D_" + append_str, "wb") as f:
            torch.save(D_.state_dict(), f)
    if D_solver:
        with open(path + "/D_optim_" + append_str, "wb") as f:
            torch.save(D_solver.state_dict(), f)


# Other functions
def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean_euclidean_error(a, b):
    mea = np.linalg.norm(np.abs(a - b), axis=2, ord=None)
    non_batch_axis = tuple(range(1, len(mea.shape)))
    mea_batch = np.mean(mea, axis=non_batch_axis)
    return mea_batch


def randn(mu, sigma):
    return sigma * np.random.randn() + mu


def get_com_occupancy_map(om):
    x, y, z = om.nonzero()
    return np.asarray([np.mean(x), np.mean(y), np.mean(z)], dtype=np.int)


def random_remove(sample, cube_len, min_remove_width=10, max_remove_width=50,
                  min_remove_voxels=1000, weighted=False, gauss_var=50, rotz=None,
                  min_rotate_deg=-75, max_rotate_deg=75,
                  others=None, args=None):
    """
    Randomly remove a part from the shape. If [out of bad luck] nothing was removed, repeat until something is removed.
    x = anteroposterior,     y = lateral,     z = vertical
    :param sample:
    :param cube_len:
    :param min_remove_width:
    :param max_remove_width:
    :param min_remove_voxels:
    :param weighted:
    :param gauss_var:
    :param rotz:
    :param min_rotate_deg:
    :param max_rotate_deg:
    :param others:
    :param args:
    :return:
    """
    while True:

        com = get_com_occupancy_map(sample)
        width = int(np.random.uniform(min_remove_width, max_remove_width))
        end_pos = com[0]

        if rotz is None:
            rotz = np.random.uniform(min_rotate_deg, max_rotate_deg)
        rotx = np.random.uniform(-5, +5)
        roty = np.random.uniform(-5, +5)

        box = np.zeros((cube_len, cube_len, cube_len), dtype=np.bool)
        box[int(end_pos - width / 2):int(end_pos + width / 2), com[1]:, :] = 1
        box = rotate(box, rotz, axes=(0, 1), reshape=False, mode='constant', cval=0, prefilter=False, order=0)
        box = rotate(box, rotx, axes=(0, 2), reshape=False, mode='constant', cval=0, prefilter=False, order=0)
        box = rotate(box, roty, axes=(1, 2), reshape=False, mode='constant', cval=0, prefilter=False, order=0)

        box_not = np.logical_xor(box, 1)  # not box
        box_not = box_not.astype(dtype=np.float32)
        box = box.astype(dtype=np.float32)

        sample = sample.astype(dtype=np.float32)
        sample_removed = sample * box
        sample_kept = sample * box_not

        if np.sum(sample_removed) > min_remove_voxels:
            # make sure ends of the shape are not cut
            width2 = width + 4
            box2_d = np.zeros((cube_len, cube_len, cube_len), dtype=np.bool)
            box2_d[int(end_pos - width2 / 2):int(end_pos - width / 2), com[1]:, :] = 1
            box2_d = rotate(box2_d, rotz, axes=(0, 1), reshape=False, mode='constant',
                            cval=0, prefilter=False, order=0)

            box2_u = np.zeros((cube_len, cube_len, cube_len), dtype=np.bool)
            box2_u[int(end_pos + width / 2):int(end_pos + width2 / 2), com[1]:, :] = 1
            box2_u = rotate(box2_u, rotz, axes=(0, 1), reshape=False, mode='constant',
                            cval=0, prefilter=False, order=0)

            boundary_pts_d = sample_kept * box2_d
            pts_x_d, pts_y_d, pts_z_d = boundary_pts_d.nonzero()

            boundary_pts_u = sample_kept * box2_u
            pts_x_u, pts_y_u, pts_z_u = boundary_pts_u.nonzero()

            if pts_x_d.shape[0] == 0 or pts_x_u.shape[0] == 0:
                # One of the boundaries is empty --> reject sample
                continue

            mean_d = np.asarray([pts_x_d.mean(), pts_y_d.mean(), pts_z_d.mean()], dtype=np.int)
            mean_u = np.asarray([pts_x_u.mean(), pts_y_u.mean(), pts_z_u.mean()], dtype=np.int)
            removed_mean = ((mean_d + mean_u) / 2).astype(np.int)

            # Get center of cut
            if weighted:
                scale = gauss_var
                gauss = multivariate_normal.pdf(np.mgrid[
                                                0:cube_len,
                                                0:cube_len,
                                                0:cube_len
                                                ].reshape(3, -1).transpose(),
                                                mean=removed_mean,
                                                cov=np.eye(3) * scale)
                # Normalize the max value to 1
                gauss /= gauss.max()
                gauss = gauss.reshape((cube_len, cube_len, cube_len))

                # set the gauss value of the main target part to 1
                gauss[box == True] = 1
            else:
                gauss = np.ones((cube_len, cube_len, cube_len))

            break

    # Cutting the same out of others
    others_weight = np.array([0] * others.shape[0])

    if others.shape[0] > 0:
        com_removed = get_com_occupancy_map(sample_removed)
        others = others.astype(dtype=np.float32)

        others_weight = []
        for i in range(others.shape[0]):
            others[i] = others[i] * box
            com_other = get_com_occupancy_map(others[i])

            translate = (com_removed - com_other).astype(dtype=np.int)
            # print('translate', translate)
            others[i] = np.roll(others[i], translate, axis=(0, 1, 2))
            d = dice_single_sample(others[i], sample_removed)
            others_weight.append(d)

            if i == others.shape[0] - 1:
                assert translate.sum() == 0

        gauss = np.tile(np.expand_dims(gauss, axis=0), (others.shape[0], 1, 1, 1))
        box = np.tile(np.expand_dims(box, axis=0), (others.shape[0], 1, 1, 1))

    return sample_kept, sample_removed, box, gauss, others, others_weight


def inject_summary(summary_writer, tag, value, step):
    import tensorflow as tf
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step=step)