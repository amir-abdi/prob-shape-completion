import os
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler

from common.utils import make_hyparam_string_gen, save_new_pickle, read_pickles_args, save_voxel_plot
from common.utils import var_or_cuda, get_data_loaders
from common.utils import dice, loss_function
from common.torch_utils import dice_torch
import cvae_shape


def train(args):
    log_param = make_hyparam_string_gen(args)

    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf
        summary_writer = tf.summary.FileWriter(os.path.join(args.output_dir, args.tb_log_dir, log_param))

        def inject_summary(summary_writer, tag, value, step):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary

    # Prepare datasets
    dset_loaders_train, dset_loaders_valid = get_data_loaders(args.data_path, labels_path=None, args=args)

    # model define
    model = getattr(cvae_shape, args.model_name)
    net = model._G(args)

    if torch.cuda.is_available():
        print("using cuda")
        net = net.cuda(args.gpu[0])
        if args.multi_gpu:
            raise NotImplementedError

    G_solver = optim.Adam(net.parameters(), lr=args.g_lr, betas=args.beta, weight_decay=args.weight_decay)

    if args.lrsh is not None:
        # todo(amirabdi): different args are needed for different schedulers
        G_scheduler = getattr(lr_scheduler, args.lrsh)(G_solver, gamma=args.lr_gamma_g)  # 0.7

    loss_fn = args.gen_g_loss[0]
    criterion_reconst = loss_function(loss_fn)

    # load saved models
    read_pickles_args(args, net, G_solver, net.vnet)

    if args.use_wandb:
        import wandb
        wandb.watch(net)

    try:
        iteration = G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][-1]]['step']
        epoch = int(iteration / int(dset_loaders_train.__len__() / args.batch_size))
        print('Continuing from iteration:{} epoch:{}'.format(iteration, epoch))
    except:
        epoch = 0
        iteration = 0
        print('Start training from scratch')

    num_view_samples = min(args.num_view_samples_per_batch, args.batch_size)

    if args.gen_g_loss[0] == 'weighted_dice':
        # todo: if implemented, remove the *box before the loss calculation
        raise NotImplementedError

    print("[------ Training ------]")
    while epoch < args.num_epochs:
        net.train(True)
        torch.cuda.empty_cache()

        # ones_weights = var_or_cuda(torch.ones((args.batch_size, 1)))
        for i, batch in enumerate(dset_loaders_train):
            if args.random_remove:
                shape_kept, shape_removed, box, gauss, others, others_weight = batch

                if shape_kept.size()[0] != int(args.batch_size):
                    continue

                shape_kept = var_or_cuda(shape_kept).unsqueeze(dim=1)
                shape_removed = var_or_cuda(shape_removed).unsqueeze(dim=1)
                box = var_or_cuda(box)
                gauss = var_or_cuda(gauss)
                others = var_or_cuda(others)
                others_weight = var_or_cuda(others_weight)

                others_weight_norm = others_weight / torch.sum(others_weight, dim=1, keepdim=True)

                generated_shape3d, kl_loss = net(shape_kept, dice_latent=others_weight, target=others)

                # TODO: not sure whether to multiply with *box before or after loss
                if loss_fn == 'variational_dice':
                    reconst_loss = criterion_reconst(generated_shape3d * box,
                                                     others,
                                                     others_weight_norm)
                elif loss_fn == 'variational_weighted_dice':
                    reconst_loss = criterion_reconst(generated_shape3d,
                                                     others + shape_kept,
                                                     others_weight_norm,
                                                     gauss)
                elif loss_fn == 'weighted_dice':
                    reconst_loss = criterion_reconst(generated_shape3d,
                                                     shape_kept + shape_removed,
                                                     gauss)
                else:
                    reconst_loss = criterion_reconst(generated_shape3d * box,
                                                     shape_removed)

                generated_shape3d = (generated_shape3d * box).mean(dim=1, keepdim=False)

                loss = reconst_loss * 1 + kl_loss * args.kl_gamma

                # for logging and visualization
                target_shape = shape_removed
            else:
                raise NotImplementedError

            loss.backward()

            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.prior.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(net.posterior.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

            # if current batch_size doesn't fit in memory, accumulate over multiple samples
            if i % args.batch_size_acml == 0:
                G_solver.step()
                net.zero_grad()
            else:
                continue

            if 'dice' in args.gen_g_loss[0]:
                dice_scores_torch = 1 - reconst_loss  # dice_torch(generated_shape3d, target_shape)
            else:
                dice_scores_torch = dice_torch(generated_shape3d, target_shape)

            try:
                iteration = \
                    G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][-1]]['step']
            except:
                iteration += 1

            # =============== save generated train images ===============#
            if iteration % args.image_save_step == 0:
                image_path = os.path.join(args.output_dir, args.image_dir + "_train", log_param)

                generated_samples = generated_shape3d.cpu().data[:num_view_samples].squeeze().numpy()
                real_samples = target_shape.cpu().data[:num_view_samples].squeeze().numpy()

                if len(real_samples.shape) == 3:
                    real_samples = np.expand_dims(real_samples, axis=0)
                    generated_samples = np.expand_dims(generated_samples, axis=0)

                dice_scores_str = [str(d) for d in dice(generated_samples, real_samples)]
                all_samples = np.concatenate((real_samples, generated_samples), axis=0)

                if not os.path.exists(image_path):
                    os.makedirs(image_path)

                save_voxel_plot(all_samples, image_path, args, str(iteration),
                                titles=dice_scores_str * 2,
                                elevation=90)
                print('Train images saved in {} with iteration {}'.format(image_path, iteration))

            # =============== logging iterations ===============#
            if iteration % args.log_step == 0:
                lr = G_solver.state_dict()['param_groups'][0]["lr"]
                info = {
                    'Epoch': epoch,
                    'iter': iteration,
                    'loss/reconst_loss': reconst_loss.data.item(),
                    'loss/kl_loss': kl_loss.data.item(),
                    'loss/loss': loss.data.item(),
                    'estimated_dice': dice_scores_torch.data.item(),
                    'lr': lr
                }
                if args.use_tensorboard:
                    log_save_path = os.path.join(args.output_dir, args.tb_log_dir, log_param)
                    if not os.path.exists(log_save_path):
                        os.makedirs(log_save_path)
                    for tag, value in info.items():
                        inject_summary(summary_writer, tag, value, str(iteration))
                    summary_writer.flush()

                if args.use_wandb:
                    wandb.log(info)

                print(
                    'Epoch:{}, Iter:{}, '
                    'loss: {:.4}, reconst_l: {:.4}, kl_l:{:.4}, '
                    'G_lr:{:.4}, dice:{:.3}'.format(
                        epoch, iteration,
                        loss.data.item(), reconst_loss.data.item(),
                        kl_loss.data.item(),
                        lr, dice_scores_torch.data.item()
                    ))

        # =============== save model as pickle ===============#
        if (epoch + 1) % args.pickle_epoch == 0:
            pickle_save_path = os.path.join(args.output_dir, args.pickle_dir, log_param)
            save_new_pickle(pickle_save_path, str(iteration), net, G_solver, iter_append=args.save_by_iter)
            print('Model saved in', pickle_save_path)

        # =============== validation ===============#
        if (epoch + 1) % args.valid_epoch == 0:
            print("[------ Validation ------]")
            net.eval()
            torch.cuda.empty_cache()
            image_path = os.path.join(args.output_dir, args.image_dir + "_valid", log_param)
            valid_dice_scores = np.array([], dtype=np.float)

            with torch.no_grad():
                for j, valid_batch in enumerate(dset_loaders_valid):

                    if args.random_remove:
                        shape_kept, shape_removed, box, _, _, _ = valid_batch
                        shape_kept = var_or_cuda(shape_kept).unsqueeze(dim=1)
                        shape_removed = var_or_cuda(shape_removed).unsqueeze(dim=1)
                        box = var_or_cuda(box)

                        latent = var_or_cuda(
                            torch.as_tensor(np.random.randn(shape_kept.size(0), 1, args.z_size) * 0, dtype=torch.float))
                        generated_shape3d = net(shape_kept, combine=True, prior_post_latent=latent).squeeze()
                        generated_shape3d = generated_shape3d * (box[:, 0].squeeze())
                        target_shape = shape_removed
                    else:
                        target_shape = valid_batch
                        generated_shape3d = net(target_shape).squeeze()

                    if target_shape.size()[0] != int(args.batch_size):
                        continue

                    generated_samples_valid = generated_shape3d.cpu().data.squeeze().numpy()
                    real_samples_valid = target_shape.cpu().data.squeeze().numpy()

                    valid_dice_scores_batch = dice(generated_samples_valid, real_samples_valid)
                    print('valid batch:{}, batch dices:{}'.format(j, valid_dice_scores_batch))
                    valid_dice_scores_batch_str = [str(f) for f in valid_dice_scores_batch]
                    valid_dice_scores = np.concatenate((valid_dice_scores, valid_dice_scores_batch))

                    valid_samples_to_save = np.concatenate((real_samples_valid[:num_view_samples],
                                                            generated_samples_valid[:num_view_samples]),
                                                           axis=0)
                    save_voxel_plot(valid_samples_to_save,
                                    image_path, args, str(iteration) + '_' + str(j),
                                    titles=valid_dice_scores_batch_str[:num_view_samples] * 2,
                                    mode='valid'
                                    )
            print('Validation images saved {} with iteration {}'.format(image_path, iteration))

            print(valid_dice_scores)
            valid_dice = valid_dice_scores.mean()
            valid_info = {"valid_dice": valid_dice}
            print('valid dice:{:.4}'.format(valid_dice))
            if args.use_wandb:
                wandb.log(valid_info)
            if args.use_tensorboard:
                for tag, value in valid_info.items():
                    inject_summary(summary_writer, tag, value, str(iteration))
                summary_writer.flush()
            print("[------ Training ------]")

        # =============== set learning rate ===============#
        if args.lrsh:
            try:
                G_scheduler.step(epoch)
            except Exception as e:
                print("fail lr scheduling", e)

        if args.use_wandb:
            wandb.log()
        epoch += 1
    print("[---- Training complete ----]")
