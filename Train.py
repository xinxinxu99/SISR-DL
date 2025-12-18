### MCNet-DL builds on the Mixed 2D/3D Convolutional Network (MCNet) originally proposed by Li et al. for hyperspectral image super-resolution.  
#- Original MCNet repository: https://github.com/qianngli/MCNet/tree/master  
#- Reference: Li, Q., Wang, Q., & Li, X. (2020). *Mixed 2D/3D convolutional network for hyperspectral image super-resolution*. Remote Sensing, 12(10), 1660.

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from Option import opt
from Model import MCNet
from data_utils import TrainsetFromFolder, ValsetFromFolder
from Metrics import PSNR
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy.io as sio

# Tag dataset+scale
tag = f"{opt.datasetName}_X{opt.upscale_factor}"

train_losses = []
val_losses = []
val_psnrs = []
val_losses_HSI = []
val_psnrs_HSI = []
Best_PSNR_HSI = -1
Best_epoch = -1

metrics_dir = f"./SISR-DL/Model/{tag}"
os.makedirs(metrics_dir, exist_ok=True)
metrics_file = os.path.join(metrics_dir, "metrics.npy")

if os.path.exists(metrics_file):
    metrics = np.load(metrics_file, allow_pickle=True).item()
else:
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "val_psnrs": [],
        "val_losses_LR_HSI": [],
        "val_psnrs_LR_HSI": [],
    }


def main():

    if opt.show:
        os.makedirs("logs/", exist_ok=True)
        global writer
        writer = SummaryWriter(log_dir="logs")

    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.cuda.empty_cache()
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    train_set = TrainsetFromFolder(f"./SISR-DL/Dataset/{tag}/Train/")
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
    )

    val_set = ValsetFromFolder(f"./SISR-DL/Dataset/{tag}/Valid/")
    val_loader = DataLoader(
        dataset=val_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=False,
    )

    val_set_LR_HSI = ValsetFromFolder(f"./SISR-DL/HSI/{tag}/")
    val_loader_LR_HSI = DataLoader(
        dataset=val_set_LR_HSI,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=False,
    )

    model = MCNet(opt)
    criterion = nn.L1Loss()

    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    # Configuration de l'optimiseur
    optimizer = optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08
    )

    # Chargement d'un checkpoint si disponible
    if opt.resume and os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Scheduler du learning rate
    scheduler = MultiStepLR(
        optimizer, milestones=[300, 400, 500, 600, 700], gamma=0.5
    )

    torch.autograd.set_detect_anomaly(True)

    global Best_PSNR_HSI, Best_epoch

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train_loss = train(train_loader, optimizer, model, criterion, epoch)
        scheduler.step()
        val_loss, val_psnr = val(
            val_loader, model, criterion, epoch, dataset_name="Default"
        )
        val_loss_LR_HSI, val_psnr_LR_HSI = val_HSI(
            val_loader_LR_HSI,
            model,
            criterion,
            epoch,
            dataset_name="Default_LR_HSI",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_losses_HSI.append(val_loss_LR_HSI)
        val_psnrs_HSI.append(val_psnr_LR_HSI)

        metrics["train_losses"].append(train_loss)
        metrics["val_losses"].append(val_loss)
        metrics["val_psnrs"].append(val_psnr)
        metrics["val_losses_LR_HSI"].append(val_loss_LR_HSI)
        metrics["val_psnrs_LR_HSI"].append(val_psnr_LR_HSI)
        np.save(metrics_file, metrics)

        plot_losses(train_losses, val_losses, val_losses_HSI, epoch)
        plot_psnr(val_psnrs, val_psnrs_HSI, epoch)

        if epoch % 10 == 0:
            save_checkpoint(epoch, model, optimizer)

        if val_psnr_LR_HSI > Best_PSNR_HSI:
            Best_PSNR_HSI = val_psnr_LR_HSI
            Best_epoch = epoch
            save_best_checkpoint(model, optimizer)
            print(
                f"New best model saved with PSNR: {Best_PSNR_HSI:.3f} at epoch {Best_epoch}"
            )


def train(train_loader, optimizer, model, criterion, epoch):
    model.train()
    epoch_loss = 0

    for iteration, batch in enumerate(train_loader, 1):
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()
        SR = model(input)
        loss = criterion(SR, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if iteration % 1 == 0:
            print(
                "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(train_loader), loss.item()
                )
            )

        if opt.show:
            niter = epoch * len(train_loader) + iteration
            if niter % 500 == 0:
                writer.add_scalar("Train/Loss", loss.item(), niter)

    return epoch_loss / len(train_loader)


def val(val_loader, model, criterion, epoch, dataset_name="Default"):
    print(f"Phase de validation ({dataset_name})")
    model.eval()
    val_loss = 0
    val_psnr = 0

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader, 1):
            input, label = Variable(batch[0]), Variable(
                batch[1], requires_grad=False
            )
            input = input.permute(0, 2, 3, 1)
            label = label.permute(0, 2, 3, 1)
            if opt.cuda:
                input = input.cuda()
                label = label.cuda()
            SR = model(input)
            loss = criterion(SR, label)
            val_loss += loss.item()
            val_psnr += PSNR(
                SR.cpu().data[0].numpy(), label.cpu().data[0].numpy()
            )

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)

    print(
        f"Validation ({dataset_name}) Loss = {val_loss:.10f}, PSNR = {val_psnr:.3f}"
    )

    if opt.show:
        writer.add_scalar(f"Val/Loss_{dataset_name}", val_loss, epoch)
        writer.add_scalar(f"Val/PSNR_{dataset_name}", val_psnr, epoch)

    return val_loss, val_psnr


def val_HSI(val_loader, model, criterion, epoch, dataset_name="Default"):
    print(f"Phase de validation ({dataset_name})")
    model.eval()
    val_loss = 0
    val_psnr = 0

    path_end = f"./SISR-DL/HSI/{tag}/{tag}_end6MV.mat"

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader, 1):
            input, label = Variable(batch[0]), Variable(
                batch[1], requires_grad=False
            )
            input = input.permute(0, 2, 3, 1)
            label = label.permute(0, 2, 3, 1)
            if opt.cuda:
                input = input.cuda()
                label = label.cuda()

            data = sio.loadmat(path_end)
            end = data["endmembers"]
            end = torch.from_numpy(end).float()

            SR = model(input)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            SR = SR.to(device)
            end = end.to(device)

            end = end.T
            end = end.view(1, 1, 1, 103, 6)  # à adapter si besoin
            SR = SR.unsqueeze(-1)

            SR = SR[:, :, :, :6, :]
            HSI_reconstructed = torch.matmul(end, SR)
            HSI_reconstructed = HSI_reconstructed.squeeze(-1)
            loss = criterion(HSI_reconstructed, label)
            val_loss += loss.item()
            val_psnr += PSNR(
                HSI_reconstructed.cpu().data[0].numpy(),
                label.cpu().data[0].numpy(),
            )

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)

    print(
        f"Validation ({dataset_name}) Loss = {val_loss:.10f}, PSNR = {val_psnr:.3f}"
    )

    if opt.show:
        writer.add_scalar(f"Val/Loss_{dataset_name}", val_loss, epoch)
        writer.add_scalar(f"Val/PSNR_{dataset_name}", val_psnr, epoch)

    return val_loss, val_psnr


def plot_losses(train_losses, val_losses, val_losses_LR_HSI, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(
        val_losses_LR_HSI,
        label="Validation Loss LR_HSI",
        linestyle="dashed",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Train & Validation Loss")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    fig_dir = f"./SISR-DL/Model/{tag}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "loss.png"))
    plt.close()


def plot_psnr(val_psnrs, val_psnrs_LR_HSI, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(val_psnrs, label="Validation PSNR", color="red")
    plt.plot(
        val_psnrs_LR_HSI,
        label="Validation PSNR LR_HSI",
        color="blue",
        linestyle="dashed",
    )
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.legend()
    plt.grid()
    fig_dir = f"./SISR-DL/Model/{tag}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "psnr.png"))
    plt.close()


def save_checkpoint(epoch, model, optimizer):
    model_dir = f"./SISR-DL/Model/{tag}"
    os.makedirs(model_dir, exist_ok=True)
    model_out_path = os.path.join(
        model_dir, f"model_{opt.upscale_factor}_epoch_{epoch}.pth"
    )
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, model_out_path)


def save_best_checkpoint(model, optimizer):
    model_dir = f"./SISR-DL/Model/{tag}"
    os.makedirs(model_dir, exist_ok=True)
    model_out_path = os.path.join(
        model_dir, f"model_{opt.upscale_factor}_Best_epoch.pth"
    )
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()
