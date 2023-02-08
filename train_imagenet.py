from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from pipeline_imagenet import YOLOPretrainLitModel


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b',
                        '--batch-size',
                        default=32,
                        type=int,
                        help="Batch size (divided if multi-gpu)")
    parser.add_argument('-lr',
                        '--learning-rate',
                        default=3e-5,
                        type=float,
                        help="Learning rate for optimizer")
    parser.add_argument('-j',
                        '--num-workers',
                        default=4,
                        type=int,
                        help='Num. of dataloader worker processes')
    parser.add_argument('-d',
                        '--devices',
                        default=1,
                        type=int,
                        help='Num. of GPU devices to train')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Toggle FP16 training')
    parser.add_argument('--no-wandb',
                        action='store_true',
                        help='Disable wandb integration')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='Resume training with checkpoint')
    parser.add_argument('dataset_root',
                        type=str,
                        help='ImageNet2012 dataset root path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    lit_model = YOLOPretrainLitModel(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        dataset_root=args.dataset_root,
    )

    trainer = Trainer(
        logger=True if args.no_wandb else WandbLogger(
            project="yolov1-imagenet"),
        precision=16 if args.fp16 else 32,
        max_epochs=200,
        accelerator='gpu',
        devices=args.devices,
        # ddp_spawn is default ddp strategy
        # but it is not recommended
        # See: https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html
        strategy='ddp' if args.devices > 1 else None,
        callbacks=[
            # EarlyStopping('val/acc_top5', patience=5),
            ModelCheckpoint(
                dirpath='./checkpoints',
                auto_insert_metric_name=False,
                filename=
                'yolov1-backbone-epoch{epoch:04d}-step{step:06d}-val_acc_top5{val_acc_top5:.2f}',
                monitor='val/acc_top5',
                mode='max',
            )
        ])

    if args.checkpoint:
        trainer.fit(lit_model, ckpt_path=args.checkpoint)
    else:
        trainer.fit(lit_model)
    print("Training sequence done.")