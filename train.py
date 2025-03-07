import argparse
import datetime
from torch.utils.data import DataLoader
from model import *
from data import *
from utils import *
import torch.nn as nn
import torch
import time
import os
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
parser.add_argument("--decay_epoch", type=int, default=40, help="epoch from which to start lr decay")
parser.add_argument("--dataroot", type=str, default="/path/to/your/dataset/",
                    help="directory of the dataset")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--num_refinement_blocks", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu_id', type=str, default='0', help='The ID of the GPU to be used.')
parser.add_argument("--b1", type=float, default=0.5, help="beta1 for adam")
parser.add_argument("--b2", type=float, default=0.999, help="beta2 for adam")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=192, help="size of image height")
parser.add_argument("--img_width", type=int, default=160, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--checkpoint_interval", type=int, default=2,
                    help="interval between saving model checkpoints")
parser.add_argument("--lambda_id", type=float, default=100.0, help="identity loss weight")
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/name_of_your_experiment/",
                    help="path to the checkpoint folder")
parser.add_argument("--filename", type=str, default="log.txt", help="name of the log file")
parser.add_argument("--start_num_train", type=int, default=0, help="Starting number of training samples")
parser.add_argument("--end_num_train", type=int, default=850, help="End number of training samples")
parser.add_argument("--start_num_val", type=int, default=850, help="Starting number of validation samples")
parser.add_argument("--end_num_val", type=int, default=870, help="End number of validation samples")
parser.add_argument("--start_slice", type=int, default=27, help="Starting number of slice")
parser.add_argument("--end_slice", type=int, default=127, help="End number of slice")
parser.add_argument("--data_list", type=str, default="./data_list/Brats2021_selected.txt",
                    help="A file recording the filtered case IDs, shuffled. The first 850 are used for training, "
                         "20 for validation, and the last 130 for testing.")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

print(opt)

if not os.path.exists(opt.ckpt_dir):
    # Create the folder if it does not exist.
    os.makedirs(opt.ckpt_dir)
    print(f"Folder '{opt.ckpt_dir}' has been created.")
else:
    print(f"Folder '{opt.ckpt_dir}' already exists.")

with open(opt.ckpt_dir + opt.filename, 'w') as file:
    file.write(str(opt))

with open(opt.ckpt_dir + opt.filename, 'a') as file:
    file.write("\nepoch  psnr  ssim   mse   mae\n")

dataloader = DataLoader(
    LoadData(opt.dataroot, ["t1", "t2", "t1ce", "seg"], opt.data_list,
             start_num=opt.start_num_train, end_num=opt.end_num_train, start_slice=opt.start_slice,
             end_slice=opt.end_slice),  # 27， 127
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True
)

val_data = DataLoader(
    LoadData(opt.dataroot, ["t1", "t2", "t1ce", "seg"], opt.data_list,
             start_num=opt.start_num_val, end_num=opt.end_num_val, start_slice=opt.start_slice,
             end_slice=opt.end_slice),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
    drop_last=True
)

# Losses
criterion_GAN = nn.MSELoss()
criterion_identity = nn.L1Loss()

device = "cuda" if opt.use_gpu else "cpu"
print(f"Using device: {device}")

# Initialize generator and discriminator
G = TLP(num_refinement_blocks=opt.num_refinement_blocks, probability_of_losing_labels=0.5,
        probability_of_dilatation=0.9, edema_operation_frequency=2, tumor_operation_frequency=5, device=device)
D = Discriminator(opt.channels)

optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

fake = torch.zeros((opt.batch_size, opt.channels, opt.img_height // 8 - 2, opt.img_width // 8 - 2), requires_grad=False)
valid = torch.ones((opt.batch_size, opt.channels, opt.img_height // 8 - 2, opt.img_width // 8 - 2), requires_grad=False)

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

fake = fake.to(device)
valid = valid.to(device)
G = G.to(device)
D = D.to(device)
criterion_GAN.to(device)
criterion_identity.to(device)

G.apply(weights_init_normal)
D.apply(weights_init_normal)

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):

    G.train()
    D.train()

    psnr_values_train = []
    ssim_values_train = []
    mse_values_train = []
    mae_values_train = []

    for i, batch in enumerate(dataloader):
        t1_data = batch['t1'][0]
        t2_data = batch['t2'][0]
        t1ce_data = batch['t1ce'][0]
        edema_region = batch['seg'][0]
        tumor_region = batch['seg'][1]

        t1_data = t1_data.to(device)
        t2_data = t2_data.to(device)
        t1ce_data = t1ce_data.to(device)
        edema_region = edema_region.to(device)
        tumor_region = tumor_region.to(device)

        optimizer_G.zero_grad()

        fake_B = G(t1_data, t2_data, edema_region, tumor_region)

        for idx in range(opt.batch_size):
            fake_B_255 = ((np.squeeze(fake_B[idx]) + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)
            real_B_255 = ((np.squeeze(t1ce_data[idx]) + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)

            if not (fake_B_255 == real_B_255).all():
                psnr_value = psnr(real_B_255, fake_B_255)
                psnr_values_train.append(psnr_value)

            ssim_value = ssim(real_B_255, fake_B_255)
            mse_value = mse(real_B_255, fake_B_255)
            mae_value = mae(real_B_255, fake_B_255)

            ssim_values_train.append(ssim_value)
            mse_values_train.append(mse_value)
            mae_values_train.append(mae_value)

        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_D.zero_grad()

        loss_real = criterion_GAN(D(t1ce_data), valid)
        loss_fake = criterion_GAN(D(fake_B.detach()), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # ------------------
        #  Train Generator
        # ------------------

        # Identity loss
        loss_identity = criterion_identity(fake_B, t1ce_data)

        # GAN loss
        loss_GAN = criterion_GAN(D(fake_B), valid)

        # Total loss
        loss_G = loss_GAN + opt.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r\033[33m[Epoch %d/%d] [Batch %d/%d] [D loss: %4f] [G loss: %4f, adv: %4f, identity: %4f] "
                         "ETA: %s\033[0m" % (epoch + 1, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(),
                                             loss_GAN.item(), loss_identity.item(), time_left))
        sys.stdout.flush()

    # Calculate the average PSNR and SSIM for the entire training batch
    avg_psnr = np.mean(psnr_values_train)
    avg_ssim = np.mean(ssim_values_train)
    avg_mse = np.mean(mse_values_train)
    avg_mae = np.mean(mae_values_train)

    print(f"\033[31m\nTraining Average PSNR: {avg_psnr: .3f}")
    print(f"Training Average SSIM: {avg_ssim: .3f}")
    print(f"Training Average MSE: {avg_mse: .3f}")
    print(f"Training Average MAE: {avg_mae: .3f}")

    # Open a file to log the losses.
    with open(opt.ckpt_dir + opt.filename, 'a') as f:
        loss = f"Training: {epoch + 1} {avg_psnr: .3f} {avg_ssim: .3f} {avg_mse: .3f} {avg_mae: .3f}\n"
        f.write(loss)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    if (epoch + 1) % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(), opt.ckpt_dir + "G_%d.pth" % (epoch + 1))
        torch.save(D.state_dict(), opt.ckpt_dir + "D_%d.pth" % (epoch + 1))

    ####################################################
    #  Evaluation
    ####################################################

    psnr_values_val = []
    ssim_values_val = []
    mse_values_val = []
    mae_values_val = []

    print("\033[32m正在进行验证")

    G.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_data):
            t1_data = batch['t1'][0]
            t2_data = batch['t2'][0]
            t1ce_data = batch['t1ce'][0]
            edema_region = batch['seg'][0]
            tumor_region = batch['seg'][1]

            t1_data = t1_data.to(device)
            t2_data = t2_data.to(device)
            t1ce_data = t1ce_data.to(device)
            edema_region = edema_region.to(device)
            tumor_region = tumor_region.to(device)

            fake_B = G(t1_data, t2_data, edema_region, tumor_region)
            real_B = t1ce_data

            fake_B = np.squeeze(fake_B)
            real_B = np.squeeze(real_B)

            fake_B_255 = ((fake_B + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)
            real_B_255 = ((real_B + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)

            if not (fake_B_255 == real_B_255).all():
                psnr_value = psnr(real_B_255, fake_B_255)
                psnr_values_val.append(psnr_value)

            ssim_value = ssim(real_B_255, fake_B_255)
            mse_value = mse(real_B_255, fake_B_255)
            mae_value = mae(real_B_255, fake_B_255)

            ssim_values_val.append(ssim_value)
            mse_values_val.append(mse_value)
            mae_values_val.append(mae_value)

    # Calculate the average PSNR and SSIM for the entire validation batch
    avg_psnr = np.mean(psnr_values_val)
    avg_ssim = np.mean(ssim_values_val)
    avg_mse = np.mean(mse_values_val)
    avg_mae = np.mean(mae_values_val)

    print(f"Validation Average PSNR: {avg_psnr: .3f}")
    print(f"Validation Average SSIM: {avg_ssim: .3f}")
    print(f"Validation Average MSE: {avg_mse: .3f}")
    print(f"Validation Average MAE: {avg_mae: .3f}")

    # Open a file to log the losses.
    with open(opt.ckpt_dir + opt.filename, 'a') as f:
        loss = f"Validation: {epoch + 1} {avg_psnr: .3f} {avg_ssim: .3f} {avg_mse: .3f} {avg_mae: .3f}\n"
        f.write(loss)

print(opt)
