import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Random delete shape
    parser.add_argument('--remove_width_max', type=float, default=50,
                        help='Maximum value for random width of removing slice (voxels).')
    parser.add_argument('--remove_width_min', type=float, default=20,
                        help='Minimum value for random width of removing slice (voxels).')
    parser.add_argument('--max_remove_rotate', type=float, default=75,
                        help='Maximum value for random rotation of the removing slice.')
    parser.add_argument('--min_remove_rotate', type=float, default=-75,
                        help='Minimum value for random rotation of the removing slice.')

    parser.add_argument('--min_remove_voxels', type=float, default=1000,
                        help='Minimum number of voxels to be removed from the shape.')
    parser.add_argument('--random_remove', type=str2bool, default=False,
                        help='To randomly remove a part from the shape.')
    parser.add_argument('--gaussian_variance', type=float, default=50,
                        help='Variance of the Gaussian distribution used to generate the 3D weighting '
                             'matrix for the weighted metrics.')
    parser.add_argument('--num_fcomb_filters', type=int, default=2,
                        help='Number of filter in convolutional layers of FCOMB.')
    parser.add_argument('--no_convs_fcomb', type=int, default=4,
                        help='Number of convolutional blocks in FCOMB.')
    parser.add_argument('--no_convs_per_block_fcomb', type=int, default=3,
                        help='Number of convolutional layers in each block of FCOMB.')

    # Model Parameters
    parser.add_argument('--vnet_skip_conn', type=str2bool, default=True,
                        help='Include skip connections between encoding and decoding of the VNet architecture.')
    parser.add_argument('--uniform_cube', type=str2bool, default=True,
                        help='When equalizing samples, create a uniform voxel representation across the 3 dimensions.')
    parser.add_argument('--conditional_d', type=str2bool, default=False,
                        help='Use conditional prob for decoder in cvae')
    parser.add_argument('--conditional_e', type=str2bool, default=False,
                        help='Use conditional prob for encoder in cvae')
    parser.add_argument('--depth', type=int, default=6,
                        help='Number of convolutional layers in the generative model.')
    parser.add_argument('--num_fc', type=int, default=0,
                        help='Number of consecutive fully connected layers in generative 3D models.')
    parser.add_argument('--padding', type=int, default=1,
                        help='Padding for all conv layers.')
    parser.add_argument('--batch_norm', type=str2bool, default=True,
                        help='Whether to use batch normalization after conv layers.')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='L2 regularization term.')
    parser.add_argument('--net_size', type=float, default=1.0,
                        help='A constant multiplier to multiply every dimension of the G with.')
    parser.add_argument('--d_thresh', type=float, default=0.8,
                        help='for balance discriminator and generator')
    parser.add_argument('--z_size', type=int, default=200,
                        help='latent space size')
    parser.add_argument('--z_dis', type=str, default="norm", choices=["norm", "uni"],
                        help='uniform: uni, normal: norm')
    parser.add_argument('--num_landmarks', type=int, default=28,
                        help='Number of labels')
    parser.add_argument('--leak_value', type=float, default=0.2,
                        help='leaky relu')
    parser.add_argument('--obj', type=str, default="chair",
                        help='training dataset object category')
    parser.add_argument('--soft_label', type=str2bool, default=False,
                        help='using soft_label')

    # Loss
    parser.add_argument('--gen_g_loss', type=str, default='dice',
                        help='Comma separated list of reconstruction losses to use to train '
                             'the generator in generative model.'
                             'Possible values are [bce|wbce|dice|focal|l1|l2|euclidean|cyclic_*]')
    parser.add_argument('--cyclic_gamma', type=float, default=1.,
                        help='Weight of the cyclic loss with respect to the '
                             'reconstruction loss which is 1.')
    parser.add_argument('--focal_gamma', type=float, default=1.,
                        help='Gamma value for focal loss (only used if gen_g_loss==focal).')
    parser.add_argument('--kl_gamma', type=float, default=5.,
                        help='Gamma value for the KL divergence loss.')
    parser.add_argument('--variation_gamma', type=float, default=0.4,
                        help='Gamma value for the variation loss.')
    parser.add_argument('--wbce_weights', type=str, default="1,1",
                        help='Weights assigned to the 0 and 1 classes in wbce.')
    parser.add_argument('--num_targets', type=int, default=5,
                        help='Number of targets to generate for each sample in each training iteration.')
    parser.add_argument('--multi_target', type=str2bool, default=True,
                        help='Learn on multiple potential targets for each sample.')

    # Training [Learning rate, optimizer, loss, batch_size]
    parser.add_argument('--lrsh', type=str, default='ExponentialLR',
                        help='Type of learning rate scheduler. set to None to keep lr constant.')
    parser.add_argument('--lr_gamma_g', type=float, default=0.999,
                        help='Gamma to multiply with the learning rate in lrsh for generator.')
    parser.add_argument('--lr_gamma_d', type=float, default=0.999,
                        help='Gamma to multiply with the learning rate in lrsh for discriminator.')
    parser.add_argument('--num_epochs', type=float, default=1000,
                        help='max epochs')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='each batch size')
    parser.add_argument('--batch_size_acml', type=int, default=1,
                        help='Sum over gradient updates of different batches and update.')
    parser.add_argument('--g_lr', type=float, default=0.001,
                        help='generator learning rate')
    parser.add_argument('--d_lr', type=float, default=0.001,
                        help='discriminator learning rate')
    parser.add_argument('--beta', type=tuple, default=(0.5, 0.5),
                        help='beta for adam')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='beta for adam')

    # Data split train and validation
    parser.add_argument('--shuffle', type=str2bool, default=True,
                        help='Shuffle data')
    parser.add_argument('--split_ratio', type=float, default=0.84,
                        help='What percentage of data to use for training')
    parser.add_argument('--split_point', type=str, default='random',
                        help='Method to split data into train and validation [iterative|random]')

    # Data augmentation
    parser.add_argument('--augment_trans', type=str2bool, default=True,
                        help='Whether to randomly translate shapes to a maximum of augment_tran_voxels.')
    parser.add_argument('--augment_trans_voxels', type=int, default=10,
                        help='Maximum number of voxels to translate in each dimension.')
    parser.add_argument('--augment_rotate', type=str2bool, default=True,
                        help='Whether to randomly rotate shapes to a maximum of augment_rotate_degree.')
    parser.add_argument('--augment_rotate_degree', type=int, default=10,
                        help='Maximum degree to rotate in each dimension.')
    parser.add_argument('--augment_mirror', type=str2bool, default=True,
                        help='Whether to flip the 3D geometry around a given axis')
    parser.add_argument('--mirror_dim', type=int, default=0,
                        help='The dimension to use for data mirroring. Ignored if mirror is False.')

    # Path parameters
    parser.add_argument('--output_dir', type=str, default="results",
                        help='output path')
    parser.add_argument('--pickle_dir', type=str, default='pickle',
                        help='input path')
    parser.add_argument('--tb_log_dir', type=str, default='log',
                        help='for tensorboard log path save in output_dir/log_dir')
    parser.add_argument('--image_dir', type=str, default='image',
                        help='for output image path save in output_dir/image_dir+[train|valid|test]')
    parser.add_argument('--data_path', type=str, default=None, required=False,
                        help='Path to dataset, single file (e.g. npy) or directory.')
    parser.add_argument('--data_path_test', type=str, default=None, required=False,
                        help='Path to load the test dataset, single file (e.g. npy) or directory.')
    parser.add_argument('--labels_path', type=str, default=None, required=False,
                        help='Path to the labels file or directory.')
    parser.add_argument('--labels_path_test', type=str, default=None, required=False,
                        help='Path to the labels file or directory for test data.')
    parser.add_argument('--load_model_path', type=str, default=None,
                        help='Path to load the saved model weights.')
    parser.add_argument('--load_optim_path', type=str, default=None,
                        help='Path to load the saved optimizer state.')
    parser.add_argument('--load_model_path2', type=str, default=None,
                        help='Path to load the second model (e.g. cyclic loss or discriminator).')
    parser.add_argument('--load_optim_path2', type=str, default=None,
                        help='Path to load the second optimizer state.')

    # Step parameter
    parser.add_argument('--log_step', type=int, default=10,
                        help='tensorboard log save at log_step epoch')
    parser.add_argument('--image_save_step', type=int, default=200,
                        help='output image save at image_save_step epoch')
    parser.add_argument('--valid_epoch', type=int, default=20,
                        help='Check accuracy on validation set.')
    parser.add_argument('--pickle_epoch', type=int, default=20,
                        help='Save trained model at pickle_step epoch')

    # Visualization and save results
    parser.add_argument('--use_tensorboard', type=str2bool, default=False,
                        help='using tensorboard logging')
    parser.add_argument('--use_wandb', type=str2bool, default=False,
                        help='using wandb logging')
    parser.add_argument('--wandb_resume_id', type=str, default=None,
                        help='Resume wandb process with the specified id.')
    parser.add_argument('--use_matplotlib', type=str2bool, default=True,
                        help='using matplotlib to save images.')
    parser.add_argument('--use_plotly', type=str2bool, default=False,
                        help='using plotly for logging')
    parser.add_argument('--save_voxels', type=str2bool, default=True,
                        help='Save voxels and landmarks numpy arrays as a single pickle file.')
    parser.add_argument('--num_view_samples_per_batch', type=int, default=4,
                        help='How many samples of each batch to visualize in maximum.')
    parser.add_argument('--save_by_iter', type=str2bool, default=True,
                        help='Append trained model names by their training iteration')

    # Dataset parameters
    parser.add_argument('--dataset_type', type=str, default=None, choices=['numpy', 'mat', 'pickle'],
                        help='Type of dataset to load. Possible values are: numpy, mat.')
    parser.add_argument('--data_resolution', type=float, default=1,
                        help='Resolution to down-sample the dataset to. Default is 1mmm (no down-samplint).')
    parser.add_argument('--cube_len', type=int, default=140,
                        help='cube length')

    # Other parameters
    parser.add_argument('--random_seed', type=int, default=123,
                        help='Random seed for all random generators')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name of the model file name used to define the architecture.')
    parser.add_argument('--postfix', type=str, default="",
                        help='A postfix to add to every path.')
    parser.add_argument('--test', type=str2bool, default=False,
                        help='Only export a bunch of test results with no training')
    parser.add_argument('--num_test', type=int, default=1,
                        help='Number of test batches to generate and save.')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of cpu workers on the data generation.')
    parser.add_argument('--multi_gpu', type=str2bool, default=False,
                        help='Split model across 2 GPUs with D on 0 and G on 1.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device index/indices. If using multi_gpu, use a comma separated list.')
    parser.add_argument('--alg_name', type=str, default=None, required=True,
                        help='Name of the algorithm [gen3d|gan3d|landmark3d|bigen3d|cvae|probvnet].')

    args = parser.parse_args()

    # interpret GPU args
    args.gpu = str2list(args.gpu, int)
    args.wbce_weights = str2list(args.wbce_weights, float)
    args.gen_g_loss = str2list(args.gen_g_loss, str)

    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v, type):
    return [type(item) for item in v.split(',')]
