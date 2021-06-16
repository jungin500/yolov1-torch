# PyTorch-version YOLOv1
# Training main model based on pretrained model

import os
import datetime
import random

from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import *

import numpy as np

import argparse

from model import YOLOv1Pretrainer, YOLOv1
from maintrainer.dataset import VOCYOLOAnnotator, VOCYolo
from maintrainer.loss import YoloLoss
from torchsummary import summary

from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=False, action='store_true', help='Enables GPU')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--num-workers', '-p', default=0, type=int, help='num_workers (default: 0)')
    parser.add_argument('--learning-rate', '-l', default=0.01, type=float, help='Learning rate (default: 0.01)')
    parser.add_argument('--seed', '-s', default=None, type=int, help='Use deterministic algorithms and give static seeds (default: None)')
    parser.add_argument('--limit-batch', '-lb', default=False, action='store_true', help='Whether to limit 20 batch (default: False)')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Epochs to run (default: 200)')
    parser.add_argument('--continue-weight', '-c', default=None, type=str, help='load weight and continue training')
    parser.add_argument('--pretrained', '-pw', default=None, type=str, help='load pretrained classifier')
    parser.add_argument('--run-name', '-rn', default='YOLOv1Maintrainer', type=str, help='Run name (used in checkpoints and tensorboard logdir name')
    args = parser.parse_args()

    run_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_') + args.run_name
    run_name += '_LR%.6f_BS%03d_WORKERS%02d_EPOCHS%03d' % (args.learning_rate, args.batch_size, args.num_workers, args.epochs)
    if args.gpu:
        run_name += '_GPU'

    # CUDA-related stuffs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        # can't use deterministic algorithms here
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    annotator = VOCYOLOAnnotator(
        annotation_root=r'C:\Development\dataset\VOCdevkit\VOC2007\Annotations',
        image_root=r'C:\Development\dataset\VOCdevkit\VOC2007\JPEGImages'
    )

    annotations = annotator.parse_annotation()
    dataset = VOCYolo(
        annotator.labels,
        annotations,
        transform=transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4547857, 0.4349471, 0.40525291],
                std=[0.12003352, 0.12323549, 0.1392444]
            )
        ])
    )

    trainlen, validlen = int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [trainlen, validlen])

    # only for args.limit_batch
    LIMIT_BATCH_SIZE = 128

    dataloader_extras = {'shuffle': True, 'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, **dataloader_extras)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, **dataloader_extras)

    if args.pretrained:
        if not os.path.isfile(args.pretrained):
            print("Pretrained weight file %s not found!" % args.pretrained)
            exit(-1)

        checkpoint = torch.load(args.pretrained)
        # c_epoch = checkpoint['epoch'] + 1
        c_model_state_dict = checkpoint['model_state_dict']
        # c_optimizer_state_dict = checkpoint['optimizer_state_dict']
        # c_loss = checkpoint['loss']

        pretrainer = YOLOv1Pretrainer(classes=1000)
        pretrainer.load_state_dict(c_model_state_dict)
        model = YOLOv1(pretrainer).to(device).float()
        del pretrainer
    else:
        model = YOLOv1().to(device).float()

    summary(model, input_size=(3, 448, 448), batch_size=args.batch_size, device=device.type)

    criterion = YoloLoss(lambda_coord=5, lambda_noobj=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    summary_writer = SummaryWriter(log_dir=os.path.join('logs', run_name))
    total_epochs = args.epochs

    if args.continue_weight:
        if not os.path.isfile(args.continue_weight):
            print("Weight file %s not found!" % args.continue_weight)
            exit(-1)

        checkpoint = torch.load(args.continue_weight)
        c_epoch = checkpoint['epoch'] + 1
        c_model_state_dict = checkpoint['model_state_dict']
        c_optimizer_state_dict = checkpoint['optimizer_state_dict']
        c_loss = checkpoint['loss']

        model.load_state_dict(c_model_state_dict)
        optimizer.load_state_dict(c_optimizer_state_dict)
        epoch_range = range(c_epoch, total_epochs)
        print("Continuing epoch %03d" % (c_epoch, ))
    else:
        epoch_range = range(total_epochs)

    # scaler = torch.cuda.amp.GradScaler()

    epoch_val_accuracies = []
    exit_reason = 0
    for epoch in epoch_range:
        print("Learning rate set to %.4f" % (optimizer.param_groups[0]['lr']))

        summary_writer.add_scalar('Hyperparameters/Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        tr = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=160,
                  desc='[Epoch %04d/%04d] Spawning Workers' % (epoch + 1, total_epochs))

        accuracies = []
        losses = []
        model.train()
        for i, (image, label) in tr:
            if args.limit_batch and i > LIMIT_BATCH_SIZE:
                break
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # with torch.cuda.amp.autocast():
            output = model(image)
            loss = criterion(output, label)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                # accuracy = torch.mean(torch.eq(torch.argmax(output, dim=1), label).int().float())
                # accuracies.append(accuracy.item())
                losses.append(loss.item())
                summary_writer.add_scalar('Training/Batch Loss', losses[-1], epoch * len(train_dataloader) + i)
                # summary_writer.add_scalar('Training/Batch Accuracy', accuracies[-1], epoch * len(train_dataloader) + i)

                tr.set_description(
                    "[Epoch %04d/%04d][Image Batch %04d/%04d] Training Loss: %.4f Training Accuracy: %.4f" %
                    (epoch + 1, total_epochs, i, len(train_dataloader), np.mean(losses), np.mean(accuracies)))

                if np.isnan(np.mean(losses)):
                    break

        if not args.limit_batch:
            train_loss_value = np.mean(losses)
            train_accuracy_value = np.mean(accuracies)

            summary_writer.add_scalar('Training/Epoch Loss', train_loss_value, epoch)
            summary_writer.add_scalar('Training/Epoch Accuracy', train_accuracy_value, epoch)

        if np.isnan(np.mean(losses)):
            print("Exiting training due to NaN Loss ...")
            exit_reason = -1
            break

        if not args.limit_batch:
            vl = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), ncols=160,
                      desc='[Epoch %04d/%04d] Spawning Workers' % (epoch + 1, total_epochs))

            accuracies = []
            top_5_accuracies = []
            losses = []
            model.eval()
            for i, (image, label) in vl:
                if args.limit_batch and i > LIMIT_BATCH_SIZE:
                    break
                if device.type != label.device.type:
                    image = image.to(device)
                    label = label.to(device)

                output = model(image)
                loss = criterion(output, label)
                # accuracy = torch.mean(torch.eq(torch.argmax(output, dim=1), label).int().float())
                # top_5_accracy = torch.mean(torch.any(torch.eq(torch.argsort(output, dim=1)[:, -5:], label.unsqueeze(1).repeat(1, 5)), 1).int().float())

                # accuracies.append(accuracy.item())
                # top_5_accuracies.append(top_5_accracy.item())
                losses.append(loss.item())

                vl.set_description("[Epoch %04d/%04d][Image Batch %04d/%04d] Validation Loss: %.4f, Accuracy: %.4f, Top-5 Accuracy: %.4f" % (
                                   epoch + 1, total_epochs, i, len(valid_dataloader), np.mean(losses), np.mean(accuracies), np.mean(top_5_accuracies)))

            for i in range(len(losses)):
                summary_writer.add_scalar('Validation/Batch Loss', losses[i], epoch * len(valid_dataloader) + i)
                # summary_writer.add_scalar('Validation/Batch Accuracy', accuracies[i], epoch * len(valid_dataloader) + i)
                # summary_writer.add_scalar('Validation/Batch Top-5 Accuracy', top_5_accuracies[i], epoch * len(valid_dataloader) + i)

            val_loss_value = np.mean(losses)
            val_acc_value = np.mean(accuracies)
            val_acc_t5_value = np.mean(top_5_accuracies)

            summary_writer.add_scalar('Validation/Epoch Loss', val_loss_value, epoch)
            # summary_writer.add_scalar('Validation/Epoch Accuracy', val_acc_value, epoch)
            # summary_writer.add_scalar('Validation/Epoch Top-5 Accuracy', val_acc_t5_value, epoch)

            if not epoch_val_accuracies or np.max(epoch_val_accuracies) < val_acc_value:
                if not os.path.isdir('.checkpoints'):
                    os.mkdir('.checkpoints')
                save_filename = os.path.join('.checkpoints',
                                             '%s-epoch%04d-train_loss%.6f-val_loss%.6f-val_acc%.6f-val_acct5%.6f.zip' % (
                                             run_name, epoch, train_loss_value, val_loss_value, val_acc_value, val_acc_t5_value))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_filename)
                print("[Epoch %04d/%04d] Saved checkpoint %s" % (epoch + 1, args.epochs, save_filename))

            epoch_val_accuracies.append(val_acc_value)

        if exit_reason != 0:
            exit(-1)
