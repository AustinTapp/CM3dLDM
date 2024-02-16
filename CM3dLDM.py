# Austin Tapp
# Children's National Hospital
# 2023
# Adapted from https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/3d_ldm/3d_ldm_tutorial.py

import os
import shutil
import tempfile
import resource
import time
import sys
import pdb
import gc

import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import torchvision
from tqdm import tqdm
from monai import transforms
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from monai.losses import SSIMLoss
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from sklearn.linear_model import LogisticRegression

from generative.inferers import LatentDiffusionInferer, DiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

obj = None
gc.collect()
torch.cuda.empty_cache()

print_config()
plt.close()

# for reproducibility purposes set a seed
set_determinism(42)

# If using Windows subsystem for Linux, the below lines are required (needed for WSL2)
torch.multiprocessing.set_sharing_strategy('file_system')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

#export DATA=/home/data/SynthRad/brain_train_restruct/images

root_dir = os.environ.get("DATA")
MR = sorted(glob.glob(root_dir + "/mri/*.nii.gz"))
CT = sorted(glob.glob(root_dir + "/ct/*.nii.gz"))
print("Working directory is:", os.getcwd())

batch_size = 1
#scale epochs, output, etc. Scale of 1 is 'standard'

run_scale = 1
n_epochs = int(1000 * run_scale)
print(f"Scale is {run_scale}\n Total number of AE epochs is {n_epochs}")

adv_weight = 0.1  # 0.05?
perceptual_weight = 1  # 0.001; 1.0
kl_weight = 1e-9  # 1e-6; 1e-9

# Provide the path to your model if you would like to resume training
AE_model_path = None
UNet_model_path = None

def check_tensor(tensor):
    # eradicate negative infinities, positive infinities, NaN, or negative zeros.
    has_neg_inf = torch.any(torch.isinf(tensor) & (tensor < 0)).item()
    has_pos_inf = torch.any(torch.isinf(tensor) & (tensor > 0)).item()
    has_nan = torch.isnan(tensor).any().item()
    has_neg_zero = torch.any(torch.signbit(tensor) & (tensor == 0)).item()
    return int(has_neg_inf or has_pos_inf or has_nan or has_neg_zero)


#change Resized to maximum feasible image given hardware resources (i.e. GPU memory limits)
#alternatively, RandSpatialCropd is a great option
image_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=(240, 240, 240), mode='empty', allow_missing_keys=False),
        transforms.CropForegroundd(keys=["image", "label"], source_key="label"),
        transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(224, 224, 224)),
        transforms.Resized(keys=["image", "label"], spatial_size=(112, 112, 112), mode=["area", "area"]),
        transforms.ScaleIntensityRangePercentilesd(keys=["image", "label"], lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

train_data = []
for idx, path in enumerate(MR):
    image_path = MR
    label_path = CT

    data_point = {"image": image_path[idx], "label": label_path[idx]}
    train_data.append(data_point)

num_images = len([data["image"] for data in train_data])
label_images = len([data["label"] for data in train_data])

print("Size of 'image' list:", num_images)
print("Size of 'label' list:", label_images)

train_proportion = 0.8
train_size = int(train_proportion * num_images)
validation_size = num_images - train_size

print("Size of training dataset:", train_size)
print("Size of validation dataset:", validation_size)

# Split the dataset into train and validation subsets
train_subset, validation_subset = train_data[:train_size], train_data[train_size:]

# Create Dataset instances for train and validation subsets
train_ds = Dataset(data=train_subset, transform=image_transforms)
validation_ds = Dataset(data=validation_subset, transform=image_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=8,
                          drop_last=True, persistent_workers=True, shuffle=True)

val_loader = DataLoader(validation_ds, batch_size=batch_size, num_workers=8,
                               drop_last=True, persistent_workers=True, shuffle=True)

check_data = first(train_loader)

image = check_data["image"]
label = check_data["label"]
print(f"image shape: {image.shape}", f"label shape: {label.shape}")

if __name__ == "__main__":
    check_data = first(train_loader)
    idx = 0

    # input dataloader check
    plt.clf()
    mr = image[idx, 0].detach().cpu().numpy()
    ct = label[idx, 0].detach().cpu().numpy()

    fig, axs = plt.subplots(nrows=2, ncols=3)
    for ax in axs.flatten():
        ax.axis("off")
    axs[0, 0].imshow(mr[..., mr.shape[2] // 2], cmap="gray")  # first row, first column
    axs[0, 1].imshow(mr[:, mr.shape[1] // 2, ...], cmap="gray")  # first row, second column
    axs[0, 2].imshow(mr[mr.shape[0] // 2, ...], cmap="gray")  # first row, third column
    axs[1, 0].imshow(ct[..., ct.shape[2] // 2], cmap="gray")
    axs[1, 1].imshow(ct[:, ct.shape[1] // 2, ...], cmap="gray")
    axs[1, 2].imshow(ct[ct.shape[0] // 2, ...], cmap="gray")
    plt.savefig("Baseline_images.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    )
    autoencoder.to(device)

    if AE_model_path is not None:
        AE_path = os.path.join(os.getcwd(), AE_model_path)
        autoencoder.load_state_dict(torch.load(AE_path))
        print("------AE successfully loaded------")

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
    discriminator.to(device)

    #change loss to use MSE instead of MAE (L1), if desired --> be sure to change throughout the rest of the script
    l1_loss = L1Loss()
    #l2_loss = MSELoss()
    structuralsim_loss = SSIMLoss(spatial_dims=3)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.25)
    loss_perceptual.to(device)

    def KL_loss(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-06)

    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-06)

    #if continuing training, it is best to set warm_up to -1, however this is not required
    n_epochs = int(1000 * run_scale)
    autoencoder_warm_up_n_epochs = 5 #-1 for cont
    val_interval = int(n_epochs / (100 * run_scale))
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    best_val_recon_epoch_loss = 1e8

    print(f"Running {n_epochs} AE epochs with {autoencoder_warm_up_n_epochs} epoch warmup delay \
    and a validation interval of {val_interval}")

    loss_g_hold = np.array([1])

    # --- Begin Autoencoder Training ---

    for epoch in range(n_epochs):
        plt.close('all')
        autoencoder.train()
        discriminator.train()
        KL_epoch_loss = 0
        ssim_epoch_loss = 0
        L1_epoch_loss = 0
        #L2_epoch_loss = 0
        p_epoch_loss = 0
        recon_epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=300)
        progress_bar.set_description(f"Train epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            label = batch["label"].to(device)
            training_file_name = images._meta['filename_or_obj'][0].split('/')[-1]
            eps = 1e-6

            # Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            reconstruction_avg = torch.mean(reconstruction)
            z_mu_avg = torch.mean(z_mu)
            z_sigma_avg = torch.mean(z_sigma)

            check_reconstruction = check_tensor(reconstruction)
            check_z_mu = check_tensor(z_mu)
            check_z_sigma = check_tensor(z_sigma)
            check_image = check_tensor(images)
            check_label = check_tensor(label)
            if check_reconstruction == 1:
               print(f"!!reconstruction nan!! in step {step}")
               reconstruction = torch.nan_to_num(reconstruction)
               reconstruction_avg = torch.mean(reconstruction)
            if check_z_mu == 1:
                print(f"!!z_mu nan!! in step {step}")
                z_mu = torch.nan_to_num(z_mu)
                z_mu_avg = torch.mean(z_mu)
            if check_z_sigma == 1:
                print(f"!!z_sigma nan!! in step {step}")
                z_sigma = torch.nan_to_num(z_sigma)
                z_sigma = torch.mean(z_sigma)
            if check_image == 1:
                print(f"!!image nan!! in step {step}")
                image = torch.nan_to_num(image)
            if check_label == 1:
                print(f"!!label nan!! in step {step}")
                label = torch.nan_to_num(label)

            kl_loss = KL_loss(z_mu, z_sigma) #+eps
            ssim_loss = structuralsim_loss(reconstruction.float(), label.float()) #+eps
            L1_loss = l1_loss(reconstruction.float(), label.float()) #+eps
            #L2_loss = l2_loss(reconstruction.float(), label.float()) #+eps
            #L1_loss = l1_loss(reconstruction.contiguous(), label.contiguous()) #+eps
            p_loss = loss_perceptual(reconstruction.float(), label.float()) #+eps
            loss_g = L1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + ssim_loss #+eps
            #loss_g = L2_loss + kl_weight * kl_loss + perceptual_weight * p_loss + ssim_loss #+eps
            loss_g_hold = loss_g

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                check_logits_fake = check_tensor(logits_fake)
                if check_logits_fake == 1:
                    print(f"!!logits_gen nan!! in step {step}")
                    logits_fake = torch.nan_to_num(logits_fake)
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            #if loss nan no backprop
            if not torch.isnan(loss_g):
                loss_g.backward()
                optimizer_g.step()
            else:
                print("loss_g is nan, setting to previous")
                loss_g = loss_g_hold

            if epoch > autoencoder_warm_up_n_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                check_logits_fake = check_tensor(logits_fake)
                if check_logits_fake == 1:
                    print("!!logits_fake_discrim!!")
                    logits_fake = torch.nan_to_num(logits_fake)
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

                logits_real = discriminator(label.contiguous().detach())[-1]
                check_logits_real = check_tensor(logits_real)
                if check_logits_real == 1:
                    print("!!logits_real_discrim!!")
                    logits_real = torch.nan_to_num(logits_real)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                # if loss nan no backprop
                if not torch.isnan(loss_d):
                    loss_d.backward()
                    optimizer_d.step()
                else:
                    print("loss_d is nan, setting to previous")
                    loss_g = loss_g_hold

            #torch.autograd.set_detect_anomaly(True)

            KL_epoch_loss += kl_loss.item()
            ssim_epoch_loss += ssim_loss.item()
            L1_epoch_loss += L1_loss.item()
            #L2_epoch_loss += L2_loss.item()
            p_epoch_loss += p_loss.item()
            recon_epoch_loss += loss_g.item()

            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "File": training_file_name,
                    "KL_loss": KL_epoch_loss / (step + 1),
                    "ssim_loss": ssim_epoch_loss / (step + 1),
                    "L1_loss": L1_epoch_loss / (step + 1),
                    #"L2_loss": L2_epoch_loss / (step + 1),
                    "p_loss": p_epoch_loss / (step + 1),
                    "Total_loss": recon_epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                    "recon avg": round(reconstruction_avg.item(), 4),
                    "z_mu avg": round(z_mu_avg.item(), 4),
                    "z_sig avg": round(z_sigma_avg.item(), 4),
                }
            )
        epoch_recon_loss_list.append(recon_epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

        # Validation intervals save a figure at each iteration
        if epoch % val_interval == 0:
            autoencoder.eval()
            discriminator.eval()
            plt.close('all')
            val_KL_epoch_loss = 0
            val_ssim_epoch_loss = 0
            val_L1_epoch_loss = 0
            #val_L2_epoch_loss = 0
            val_p_epoch_loss = 0
            val_recon_epoch_loss = 0
            val_gen_epoch_loss = 0
            val_disc_epoch_loss = 0
            with torch.no_grad():
                progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=300)
                progress_bar.set_description(f"Val epoch {epoch}")
                for step, batch in progress_bar:
                    images = batch["image"].to(device)
                    label = batch["label"].to(device)
                    training_file_name = images._meta['filename_or_obj'][0].split('/')[-1]
                    eps = 1e-6

                    # Generator part
                    optimizer_g.zero_grad(set_to_none=True)
                    reconstruction, z_mu, z_sigma = autoencoder(images)

                    reconstruction_avg = torch.mean(reconstruction)
                    z_mu_avg = torch.mean(z_mu)
                    z_sigma_avg = torch.mean(z_sigma)

                    check_reconstruction = check_tensor(reconstruction)
                    check_z_mu = check_tensor(z_mu)
                    check_z_sigma = check_tensor(z_sigma)
                    check_image = check_tensor(images)
                    check_label = check_tensor(label)

                    if check_reconstruction == 1:
                        print(f"!!reconstruction nan!! in step {step}")
                        reconstruction = torch.nan_to_num(reconstruction)
                        reconstruction_avg = torch.mean(reconstruction)
                    if check_z_mu == 1:
                        print(f"!!z_mu nan!! in step {step}")
                        z_mu = torch.nan_to_num(z_mu)
                        z_mu_avg = torch.mean(z_mu)
                    if check_z_sigma == 1:
                        print(f"!!z_sigma nan!! in step {step}")
                        z_sigma = torch.nan_to_num(z_sigma)
                        z_sigma = torch.mean(z_sigma)
                    if check_image == 1:
                        print(f"!!image nan!! in step {step}")
                        image = torch.nan_to_num(image)
                    if check_label == 1:
                        print(f"!!label nan!! in step {step}")
                        label = torch.nan_to_num(label)

                    kl_loss = KL_loss(z_mu, z_sigma)  # +eps
                    ssim_loss = structuralsim_loss(reconstruction.float(), label.float())  # +eps
                    L1_loss = l1_loss(reconstruction.float(), label.float()) #+eps
                    #L2_loss = l2_loss(reconstruction.float(), label.float())  # +eps
                    # L1_loss = l1_loss(reconstruction.contiguous(), label.contiguous()) #+eps
                    p_loss = loss_perceptual(reconstruction.float(), label.float())  # +eps
                    loss_g = L1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + ssim_loss #+eps
                    #loss_g = L2_loss + kl_weight * kl_loss + perceptual_weight * p_loss + ssim_loss  # +eps
                    loss_g_hold = loss_g

                    if epoch > autoencoder_warm_up_n_epochs:
                        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                        check_logits_fake = check_tensor(logits_fake)
                        if check_logits_fake == 1:
                            print(f"!!logits_gen nan!! in step {step}")
                            logits_fake = torch.nan_to_num(logits_fake)

                        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                        loss_g += adv_weight * generator_loss
                        # loss_g = loss_g +eps

                    # if loss nan no backprop
                    if torch.isnan(loss_g):
                        print("loss_g is nan, setting to previous")
                        loss_g = loss_g_hold

                    if epoch > autoencoder_warm_up_n_epochs:
                        # Discriminator part
                        optimizer_d.zero_grad(set_to_none=True)

                        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                        check_logits_fake = check_tensor(logits_fake)
                        if check_logits_fake == 1:
                            print("!!logits_fake_discrim!!")
                            logits_fake = torch.nan_to_num(logits_fake)
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

                        logits_real = discriminator(label.contiguous().detach())[-1]
                        check_logits_real = check_tensor(logits_real)
                        if check_logits_real == 1:
                            print("!!logits_real_discrim!!")
                            logits_real = torch.nan_to_num(logits_real)
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                        loss_d = adv_weight * discriminator_loss

                        # if loss nan no backprop
                        if torch.isnan(loss_d):
                            print("loss_d is nan, setting to previous")
                            loss_g = loss_g_hold

                    # torch.autograd.set_detect_anomaly(True)

                    val_KL_epoch_loss += kl_loss.item()
                    val_ssim_epoch_loss += ssim_loss.item()
                    val_L1_epoch_loss += L1_loss.item()
                    #val_L2_epoch_loss += L2_loss.item()
                    val_p_epoch_loss += p_loss.item()
                    val_recon_epoch_loss += loss_g.item()

                    if epoch > autoencoder_warm_up_n_epochs:
                        val_gen_epoch_loss += generator_loss.item()
                        val_disc_epoch_loss += discriminator_loss.item()

                    progress_bar.set_postfix(
                        {
                            "File": training_file_name,
                            "KL_loss": val_KL_epoch_loss / (step + 1),
                            "ssim_loss": val_ssim_epoch_loss / (step + 1),
                            "L1_loss": val_L1_epoch_loss / (step + 1),
                            #"L2_loss": val_L2_epoch_loss / (step + 1),
                            "p_loss": val_p_epoch_loss / (step + 1),
                            "Total_loss": val_recon_epoch_loss / (step + 1),
                            "gen_loss": val_gen_epoch_loss / (step + 1),
                            "disc_loss": val_disc_epoch_loss / (step + 1),
                            "recon avg": round(reconstruction_avg.item(), 4),
                            "z_mu avg": round(z_mu_avg.item(), 4),
                            "z_sig avg": round(z_sigma_avg.item(), 4),
                        }
                    )

                    val_recon_epoch_loss += val_recon_epoch_loss
                val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
                val_recon_epoch_loss = torch.tensor(val_recon_epoch_loss).to(device)

                if val_recon_epoch_loss < best_val_recon_epoch_loss:
                    best_val_recon_epoch_loss = val_recon_epoch_loss
                    torch.save(
                        autoencoder.state_dict(),
                        os.path.join(os.getcwd(), "AE_model_best_val.pth"),)

                plt.clf()
                idx = 0
                mr = images[idx, 0].detach().cpu().numpy()
                ct = label[idx, 0].detach().cpu().numpy()
                recon = reconstruction[idx, 0].detach().cpu().numpy()

                fig, axs = plt.subplots(nrows=3, ncols=3)
                for ax in axs.flatten():
                    ax.axis("off")

                axs[0, 0].imshow(mr[..., mr.shape[2] // 2], cmap="gray")  # first row, first column
                axs[0, 1].imshow(mr[:, mr.shape[1] // 2, ...], cmap="gray")  # first row, second column
                axs[0, 2].imshow(mr[mr.shape[0] // 2, ...], cmap="gray")  # first row, third column

                axs[1, 0].imshow(ct[..., mr.shape[2] // 2], cmap="gray")  # second row, first column
                axs[1, 1].imshow(ct[:, mr.shape[1] // 2, ...], cmap="gray")  # second row, second column
                axs[1, 2].imshow(ct[mr.shape[0] // 2, ...], cmap="gray")  # second row, third column

                axs[2, 0].imshow(recon[..., mr.shape[2] // 2], cmap="gray")  # second row, first column
                axs[2, 1].imshow(recon[:, mr.shape[1] // 2, ...], cmap="gray")  # second row, second column
                axs[2, 2].imshow(recon[mr.shape[0] // 2, ...], cmap="gray")  # second row, third column
                plt.savefig(f"ReconstructedImages_Epoch_{epoch}.png")

            torch.save(
                autoencoder.state_dict(),
                os.path.join(os.getcwd(), f"AE_model_epoch_{epoch}.pth"),
                )

    torch.save(autoencoder.state_dict(), os.path.join(os.getcwd(), "AE_model_finished.pth"),)

    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()

    # --- End Autoencoder Training ---
    plt.clf()
    plt.style.use("ggplot")
    plt.title("Learning Curves", fontsize=20)
    plt.plot(epoch_recon_loss_list)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig("AllReconTrainingCurves.png")

    plt.clf()
    plt.title("Adversarial Training Curves", fontsize=20)
    plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
    plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig("AdversarialTrainingCurves.png")

    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    # --- End Autoencoder ---'''

    # --- Begin Diffusion Model ---

    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
    )
    unet.to(device)
    
    if UNet_model_path is not None:
        UNet_path = os.path.join(os.getcwd(), UNet_model_path)
        unet.load_state_dict(torch.load(UNet_path))
        print("------UNet successfully loaded------")

    scheduler_ddpm = DDPMScheduler(num_train_timesteps=2000, beta_schedule="scaled_linear", beta_start=0.0015, beta_end=0.0195)
    scheduler_ddpm.set_timesteps(num_inference_steps=2000)
    inferer_ddpm = LatentDiffusionInferer(scheduler_ddpm, scale_factor=scale_factor)

    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)
    scaler = GradScaler()

    scheduler_ddim = DDIMScheduler(num_train_timesteps=2000, beta_schedule="scaled_linear", beta_start=0.0015, beta_end=0.0195)
    scheduler_ddim.set_timesteps(num_inference_steps=2000)

    # --- Begin Diffusion UNet Training ---

    n_epochs = int(1000 * run_scale)
    epoch_loss_list = []
    val_interval = int(n_epochs / (100 * run_scale))
    best_val_UNet_epoch_loss = 1e6
    autoencoder.eval()

    for epoch in range(n_epochs):
        plt.close('all')
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=200)
        progress_bar.set_description(f"Train Epoch {epoch}")
        for step, batch in progress_bar:
            image = batch["image"].to(device)
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                with torch.no_grad():
                    z = autoencoder.encode_stage_2_inputs(image) * scale_factor

                noise_like_z = torch.randn_like(z).to(device)

                tsteps = torch.randint(0, inferer_ddpm.scheduler.num_train_timesteps,
                                       (image.shape[0],), device=image.device).long()

                prediction = inferer_ddpm(
                    inputs=image, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise_like_z,
                    timesteps=tsteps)

                noisy_z_avg = torch.mean(noise_like_z)
                pred_avg = torch.mean(prediction)

                check_noisy_z = check_tensor(noise_like_z)
                if check_noisy_z == 1:
                    print("!!noise nan!!")
                    noise_like_z = torch.nan_to_num(noise_like_z)

                check_prediction = check_tensor(prediction)
                if check_prediction == 1:
                    print("!!prediction nan!!")
                    prediction = torch.nan_to_num(prediction)

                loss = F.mse_loss(prediction.float(), noise_like_z.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1),
                                      "noise z avg": round(noisy_z_avg.item(), 4),
                                      "pred_avg avg": round(pred_avg.item(), 4)})

        epoch_loss_list.append(epoch_loss / (step + 1))

        if epoch % val_interval == 0:
            autoencoder.eval()
            unet.eval()
            val_Unet_epoch_loss = 0
            with torch.no_grad():
                progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=200)
                progress_bar.set_description(f"Val Epoch {epoch}")
                for step, batch in progress_bar:
                    image = batch["image"].to(device)
                    optimizer_diff.zero_grad(set_to_none=True)

                    with autocast(enabled=True):
                        with torch.no_grad():
                            z = autoencoder.encode_stage_2_inputs(image) * scale_factor

                        noise_like_z = torch.randn_like(z).to(device)

                        tsteps = torch.randint(0, inferer_ddpm.scheduler.num_train_timesteps, (image.shape[0],), device=image.device).long()

                        prediction = inferer_ddpm(inputs=image, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise_like_z, timesteps=tsteps)

                        noisy_z_avg = torch.mean(noise_like_z)
                        pred_avg = torch.mean(prediction)

                        check_noisy_z = check_tensor(noise_like_z)
                        if check_noisy_z == 1:
                            print("!!noise nan!!")
                            noise_like_z = torch.nan_to_num(noise_like_z)

                        check_prediction = check_tensor(prediction)
                        if check_prediction == 1:
                            print("!!prediction nan!!")
                            prediction = torch.nan_to_num(prediction)

                        val_loss = F.mse_loss(prediction.float(), noise_like_z.float())

                        progress_bar.set_postfix({"loss": val_loss.item() / (step + 1),
                                                  "noise z avg": round(noisy_z_avg.item(), 4),
                                                  "pred_avg avg": round(pred_avg.item(), 4)})

                        z_image = autoencoder.encode_stage_2_inputs(image).to(device)
                        noise = torch.randn_like(z_image).to(device)
                        z_and_noise = (z_image + noise).to(device)

                        inferer_ddim = LatentDiffusionInferer(scheduler_ddim, scale_factor=scale_factor)

                        sCT_ddpm = inferer_ddpm.sample(input_noise=noise, autoencoder_model=autoencoder,
                                                       diffusion_model=unet, scheduler=scheduler_ddpm)

                        sCT_zAndnoise_ddpm = inferer_ddpm.sample(input_noise=z_and_noise, autoencoder_model=autoencoder,
                                                                 diffusion_model=unet, scheduler=scheduler_ddpm)

                        sCT_ddim = inferer_ddim.sample(input_noise=noise, autoencoder_model=autoencoder,
                                                       diffusion_model=unet, scheduler=scheduler_ddim)

                        sCT_zAndnoise_ddim = inferer_ddpm.sample(input_noise=z_and_noise, autoencoder_model=autoencoder,
                                                                 diffusion_model=unet, scheduler=scheduler_ddim)

                        plt.clf()
                        idx = 0
                        img = image[idx, 0].detach().cpu().numpy()
                        synthetic_ddpm = sCT_ddpm[idx, 0].detach().cpu().numpy()
                        synthetic_ddim = sCT_ddim[idx, 0].detach().cpu().numpy()
                        synthetic_Znoise_ddpm = sCT_zAndnoise_ddpm[idx, 0].detach().cpu().numpy()
                        synthetic_Znoise_ddim = sCT_zAndnoise_ddim[idx, 0].detach().cpu().numpy()
                        fig, axs = plt.subplots(nrows=1, ncols=5)
                        for ax in axs:
                            ax.axis("off")
                        ax = axs[0]
                        ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
                        ax = axs[1]
                        ax.imshow(synthetic_ddpm[:, img.shape[1] // 2, ...], cmap="gray")
                        ax = axs[2]
                        ax.imshow(synthetic_ddim[:, img.shape[1] // 2, ...], cmap="gray")
                        ax = axs[3]
                        ax.imshow(synthetic_Znoise_ddpm[:, img.shape[1] // 2, ...], cmap="gray")
                        ax = axs[4]
                        ax.imshow(synthetic_Znoise_ddim[:, img.shape[1] // 2, ...], cmap="gray")
                        plt.savefig(f"diff_Unet_output_{epoch}.png")

                    val_Unet_epoch_loss += val_loss.item()
                val_Unet_epoch_loss = val_Unet_epoch_loss / (step + 1)
                val_Unet_epoch_loss = torch.tensor(val_Unet_epoch_loss).to(device)

                if val_Unet_epoch_loss < best_val_UNet_epoch_loss:
                    best_val_UNet_epoch_loss = val_Unet_epoch_loss
                    torch.save(
                        unet.state_dict(),
                        os.path.join(os.getcwd(), "dUNet_model_best_val.pth"),
                    )

            torch.save(
                unet.state_dict(),
                os.path.join(os.getcwd(), f"dUNet_model_epoch_{epoch}.pth"),
            )

    torch.save(
        unet.state_dict(),
        os.path.join(os.getcwd(), "dUNet_model_final.pth"),
    )

    plt.clf()
    plt.plot(epoch_loss_list)
    plt.title("Learning Curves", fontsize=20)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig("DiffusionUNetLearningCurves.png")

    torch.cuda.empty_cache()
