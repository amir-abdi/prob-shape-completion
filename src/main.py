import importlib
import os

from common.utils import set_random_seed
from common.parse_args import parse_args


def main(args):

    module_name = importlib.import_module(args.alg_name)
    if args.test:
        module_name.test(args)
    else:
        module_name.train(args)


if __name__ == '__main__':
    args = parse_args()
    # for arg in vars(args):
    #     print('{}:{}'.format(arg, getattr(args, arg)))

    # Override args in test mode
    if args.test:
        args.use_wandb = False
        args.shuffle = False
        args.split_point = 'iterative'
        args.split_ratio = 0
        args.augment_trans = False
        args.augment_rotate = False
        args.random_remove = False
        args.augment_mirror = False
        # args.batch_size = 4
        args.save_voxels = True

    # initialize wandb
    if args.use_wandb:
        import wandb

        resume_wandb = True if args.wandb_resume_id is not None else False
        wandb.init(config=args, resume=resume_wandb, id=args.wandb_resume_id,
                   project='ProbShapeCompletion')

    # set random seeds for np, torch, python
    set_random_seed(args.random_seed)

    main(args)
