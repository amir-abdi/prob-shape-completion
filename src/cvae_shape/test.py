import torch
from common.utils import make_hyparam_string_gen, read_pickle_path, save_voxel_plot2
from common.utils import get_data_loaders, prepare_visualization
from common.utils import dice, var_or_cuda

import os
import numpy as np
import cvae_shape as current_module


def test(args):
    log_param = make_hyparam_string_gen(args)

    # model define
    model = getattr(current_module, args.model_name)
    net = model._G(args)

    if torch.cuda.is_available():
        print("using cuda")
        net = net.cuda(args.gpu[0])

    read_pickle_path(args.load_model_path, net)

    # Prepare datasets
    print('Loading data:', args.data_path_test)
    _, dset_loaders_test = get_data_loaders(args.data_path_test, labels_path=None, args=args)

    dice_scores = np.array([], dtype=np.float)
    image_path = os.path.join(args.output_dir, args.image_dir + '_test', log_param)
    indices = dset_loaders_test.sampler.indices
    num_batches = (len(indices) // args.batch_size)
    print('Saving images in', image_path)

    net.eval()
    with torch.no_grad():
        for j, test_batch in enumerate(dset_loaders_test):
            shape_kept, shape_removed, box = test_batch[:3]
            shape_kept = var_or_cuda(shape_kept).unsqueeze(dim=1)
            shape_removed = var_or_cuda(shape_removed).unsqueeze(dim=1)
            box = var_or_cuda(box)

            if shape_kept.size(0) != args.batch_size:
                continue

            latent = var_or_cuda(torch.as_tensor(np.random.randn(shape_kept.size(0),
                                                                 args.num_targets,  # number of variations
                                                                 args.z_size), dtype=torch.float))

            generated_shape3d = net(shape_kept, combine=True, prior_post_latent=latent).squeeze()
            generated_shape3d = generated_shape3d * box

            generated_samples = generated_shape3d.cpu().data.squeeze().numpy()
            shape_removed = shape_removed.cpu().data.squeeze().numpy()
            shape_kept = shape_kept.cpu().data.squeeze().numpy()
            reconstructed = generated_samples + shape_kept

            test_dice_scores_batch = dice(generated_samples, shape_removed)
            print('batch:{}/{}, batch dices:{}'.format(j, num_batches, test_dice_scores_batch))
            test_dice_scores_batch_str = [str(f) for f in test_dice_scores_batch]
            dice_scores = np.concatenate((dice_scores, test_dice_scores_batch))
            to_visualize = prepare_visualization([shape_removed, generated_samples, shape_kept, reconstructed])
            save_voxel_plot2(to_visualize, image_path, args, str(j), titles=test_dice_scores_batch_str, mode='test')

    print('Test images saved {}'.format(image_path))
    print(dice_scores)
    test_dice = dice_scores.mean()
    print('test dice:{:.4}'.format(test_dice))

    results_path = os.path.join(image_path, "test_results_{}.csv".format(test_dice))
    num_tested = args.batch_size * num_batches
    np.savetxt(results_path, np.stack((indices[:num_tested], dice_scores), axis=-1), delimiter=",")
    print('Results saved in', results_path)
