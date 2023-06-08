import argparse
import os
import time
import logging
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/d2nerf.yaml', help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
    )
    parser.add_argument('--mesh_only', action='store_true', help='when test mode, switch on to extract mesh only without generate test images')
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')


    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--validate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--predict', action='store_true')

    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # solar config
    # thread control, might not working
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['OPENBLAS_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '16'
    n_gpus = len(args.gpu.split(','))

    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

    import datasets
    import systems
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar
    from utils.misc import load_config    
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from models.utils import cleanup

    # free cache and temporary memory of tcnn
    cleanup()

    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = os.path.join(config.get('exp_dir'))
    config.runs_dir = config.get('runs_dir') or os.path.join(config.exp_dir, config.trial_name, 'runs')
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')

    if args.test:
        if args.mesh_only:
            config.mesh_only = True
        else:
            config.mesh_only = False
    else:
        config.mesh_only = False

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)


    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None if not args.resume_weights_only else args.resume)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            LearningRateMonitor(logging_interval='step'),
            CodeSnapshotCallback(
                config.code_dir, use_version=False
            ),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=1),
            # ModelSummary(max_depth=-1)
        ]

    loggers = []
    if args.train:
        loggers += [
            TensorBoardLogger(config.runs_dir, version=config.trial_name),
            CSVLogger(config.exp_dir, name=config.trial_name, version='csv_logs')
        ]

    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        # auto_select_gpus=True,
        callbacks=callbacks,
        logger=loggers,
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy='dp',
        **config.trainer
    )

    if args.train:
        if args.resume and not args.resume_weights_only:
            trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=args.resume)
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=args.resume)
    
    # free cache and temporary memory of tcnn
    cleanup()


if __name__ == '__main__':
    main()