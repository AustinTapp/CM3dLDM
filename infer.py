import logging
import os
import warnings
import pdb
import sys
import shutil
import tempfile
import resource
import time
import gc

from typing import Optional, Sequence, Union
from functools import partial

import torch
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from monai import transforms
from monai.config import print_config
from monai.data import ThreadDataLoader, decollate_batch, list_data_collate
from monai.utils import first, set_determinism
from monai.losses import SSIMLoss
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from sklearn.linear_model import LogisticRegression

from generative.inferers import LatentDiffusionInferer, DiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

print_config()
print("Working from:", os.getcwd())
torch.cuda.empty_cache()

set_determinism(42)

# needed for WSL2
torch.multiprocessing.set_sharing_strategy('file_system')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

infer_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.SpatialPadd(keys=["image"], spatial_size=(280, 280, 280), mode='empty'),
        transforms.CropForegroundd(keys=["image"], source_key="image"),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(224, 224, 224)),
        transforms.Resized(keys=["image"], spatial_size=(112, 112, 112), mode=["area"]),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1)])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    #provide your input and data directories
    #each directory should have a folder named after a case

    data_dir = os.path.join(os.getcwd(), "SynthRad/data/mr_images")
    save_dir = os.path.join(os.getcwd(), "SynthRad/data/sCT_images")

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

    # load your AE model here
    AE_path = os.path.join(os.getcwd(), "AE_model.pth")
    autoencoder.load_state_dict(torch.load(AE_path))
    print("------AE successfully loaded------")
    print(f"[info] checkpoint {AE_path:s} loaded")

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

    Unet_path = os.path.join(os.getcwd(), "dUNet_model.pth")
    unet.load_state_dict(torch.load(Unet_path))
    print("------Unet successfully loaded------")
    print(f"[info] checkpoint {Unet_path:s} loaded")

    autoencoder.eval()
    unet.eval()

    scheduler_ddpm = DDPMScheduler(num_train_timesteps=2000, beta_schedule="scaled_linear", beta_start=0.0015, beta_end=0.0195)
    scheduler_ddpm.set_timesteps(num_inference_steps=2000)

    files = []
    folder_list = [file for file in os.listdir(data_dir) if
                   os.path.isdir(os.path.join(data_dir, file))]
    for i in range(len(folder_list)):
        image_name = str("mr.nii.gz")
        str_img = []
        str_img.append(os.path.join(data_dir, folder_list[i], image_name))
        files.append({"image": str_img})
    infer_files = files

    for i in range(len(infer_files)):
        MR_image = infer_files[i]
        pred_img_name = MR_image["image"][0].split("/")[-2]

        with torch.no_grad():
            batch_data = infer_transforms(MR_image)
            batch_data = list_data_collate([batch_data])
            infer_image = batch_data["image"].to(device)

            with autocast(enabled=True):
                with torch.no_grad():
                    z = autoencoder.encode_stage_2_inputs(infer_image).to(device)
                    scale_factor = 1 / torch.std(z)
                    print(f"---Z Scale factor is {scale_factor} ---")

                z_image = autoencoder.encode_stage_2_inputs(infer_image).to(device)
                noise = torch.randn_like(z_image).to(device)
                noise2 = torch.randn_like(z_image).to(device)
                z_and_noise = (z_image + noise).to(device)
                z_noise_2 = (z_image + noise + noise2).to(device)

                inferer_ddpm = LatentDiffusionInferer(scheduler_ddpm, scale_factor=scale_factor)

                sCT_ddpm = inferer_ddpm.sample(input_noise=noise, autoencoder_model=autoencoder,
                                               diffusion_model=unet, scheduler=scheduler_ddpm)

                sCT_ddpm.applied_operations = batch_data["image"].applied_operations
                batch_data.update({'image': sCT_ddpm})
                batch_data = [infer_transforms.inverse(i) for i in decollate_batch(batch_data)]
                synthetic_ddpm = batch_data[0]["image"]

                sCT_zAndnoise_ddpm = inferer_ddpm.sample(input_noise=z_and_noise, autoencoder_model=autoencoder,
                                                         diffusion_model=unet, scheduler=scheduler_ddpm)

                batch_data = infer_transforms(MR_image)
                batch_data = list_data_collate([batch_data])
                sCT_zAndnoise_ddpm.applied_operations = batch_data["image"].applied_operations
                batch_data.update({'image': sCT_zAndnoise_ddpm})
                batch_data = [infer_transforms.inverse(i) for i in decollate_batch(batch_data)]
                synthetic_Znoise_ddpm = batch_data[0]["image"]

                sCT_zAndnoise_2_ddpm = inferer_ddpm.sample(input_noise=z_noise_2, autoencoder_model=autoencoder,
                                                         diffusion_model=unet, scheduler=scheduler_ddpm)

                batch_data = infer_transforms(MR_image)
                batch_data = list_data_collate([batch_data])
                sCT_zAndnoise_2_ddpm.applied_operations = batch_data["image"].applied_operations
                batch_data.update({'image': sCT_zAndnoise_2_ddpm})
                batch_data = [infer_transforms.inverse(i) for i in decollate_batch(batch_data)]
                synthetic_Znoise_2_ddpm = batch_data[0]["image"]

                sCT_z_ddpm = inferer_ddpm.sample(input_noise=z_image, autoencoder_model=autoencoder,
                                                           diffusion_model=unet, scheduler=scheduler_ddpm)

                batch_data = infer_transforms(MR_image)
                batch_data = list_data_collate([batch_data])
                sCT_z_ddpm.applied_operations = batch_data["image"].applied_operations
                batch_data.update({'image': sCT_z_ddpm})
                batch_data = [infer_transforms.inverse(i) for i in decollate_batch(batch_data)]
                synthetic_Z_ddpm = batch_data[0]["image"]

                batch_data = infer_transforms(MR_image)
                batch_data = list_data_collate([batch_data])
                infer_image = batch_data["image"].to(device)
                infer_image.applied_operations = batch_data["image"].applied_operations
                batch_data.update({'image': infer_image})
                batch_data = [infer_transforms.inverse(i) for i in decollate_batch(batch_data)]
                infer_image = batch_data[0]["image"]

                img = infer_image[0].detach().cpu().numpy()
                synthetic_ddpm = synthetic_ddpm[0].detach().cpu().numpy()
                synthetic_Znoise_ddpm = synthetic_Znoise_ddpm[0].detach().cpu().numpy()
                synthetic_Znoise_2_ddpm = synthetic_Znoise_2_ddpm[0].detach().cpu().numpy()
                synthetic_Z_ddpm = synthetic_Z_ddpm[0].detach().cpu().numpy()

                ct_path = MR_image["image"][0].replace('mr.nii.gz', 'ct.nii.gz')
                if ct_path:
                    CT_image = MR_image
                    CT_image.update({'image': ct_path})
                    batch_data = infer_transforms(CT_image)
                    batch_data = list_data_collate([batch_data])
                    ct_image_tfm = batch_data["image"].to(device)
                    ct_image_tfm.applied_operations = batch_data["image"].applied_operations
                    batch_data.update({'image': ct_image_tfm})
                    batch_data = [infer_transforms.inverse(i) for i in decollate_batch(batch_data)]
                    ct = batch_data[0]["image"]

                    ct = ct[0].detach().cpu().numpy()
                else:
                    ct = img

                plt.clf()
                fig, axs = plt.subplots(nrows=3, ncols=6)
                for ax in axs.flatten():
                    ax.axis("off")
                axs[0, 0].imshow(img[..., img.shape[2] // 2], cmap="gray")
                axs[1, 0].imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
                axs[2, 0].imshow(img[img.shape[0] // 2, ...], cmap="gray")
                axs[0, 1].imshow(synthetic_ddpm[..., img.shape[2] // 2], cmap="gray")
                axs[1, 1].imshow(synthetic_ddpm[:, img.shape[1] // 2, ...], cmap="gray")
                axs[2, 1].imshow(synthetic_ddpm[img.shape[0] // 2, ...], cmap="gray")
                axs[0, 2].imshow(synthetic_Znoise_ddpm[..., img.shape[2] // 2], cmap="gray")
                axs[1, 2].imshow(synthetic_Znoise_ddpm[:, img.shape[1] // 2, ...], cmap="gray")
                axs[2, 2].imshow(synthetic_Znoise_ddpm[img.shape[0] // 2, ...], cmap="gray")
                axs[0, 3].imshow(synthetic_Znoise_2_ddpm[..., img.shape[2] // 2], cmap="gray")
                axs[1, 3].imshow(synthetic_Znoise_2_ddpm[:, img.shape[1] // 2, ...], cmap="gray")
                axs[2, 3].imshow(synthetic_Znoise_2_ddpm[img.shape[0] // 2, ...], cmap="gray")
                axs[0, 4].imshow(synthetic_Z_ddpm[..., img.shape[2] // 2], cmap="gray")
                axs[1, 4].imshow(synthetic_Z_ddpm[:, img.shape[1] // 2, ...], cmap="gray")
                axs[2, 4].imshow(synthetic_Z_ddpm[img.shape[0] // 2, ...], cmap="gray")
                axs[0, 5].imshow(ct[..., img.shape[2] // 2], cmap="gray")
                axs[1, 5].imshow(ct[:, img.shape[1] // 2, ...], cmap="gray")
                axs[2, 5].imshow(ct[img.shape[0] // 2, ...], cmap="gray")

                plt.savefig(os.path.join(save_dir, f"sct_{pred_img_name}_all_outputs.png"))

                img = img.astype(np.float32)
                img = sitk.GetImageFromArray(img)
                sitk.WriteImage(img, os.path.join(save_dir, f"MR_tfm_{pred_img_name}.nii.gz"))

                synthetic_ddpm_pred = synthetic_ddpm.astype(np.float32)
                synthetic_ddpm_pred = sitk.GetImageFromArray(synthetic_ddpm_pred)
                sitk.WriteImage(synthetic_ddpm_pred, os.path.join(save_dir, f"{pred_img_name}_ddpm.nii.gz"))

                synthetic_Znoise_ddpm_pred = synthetic_Znoise_ddpm.astype(np.float32)
                synthetic_Znoise_ddpm_pred = sitk.GetImageFromArray(synthetic_Znoise_ddpm_pred)
                sitk.WriteImage(synthetic_Znoise_ddpm_pred, os.path.join(save_dir, f"{pred_img_name}_ddpm_noised.nii.gz"))

                synthetic_Znoise_2_ddpm_pred = synthetic_Znoise_2_ddpm.astype(np.float32)
                synthetic_Znoise_2_ddpm_pred = sitk.GetImageFromArray(synthetic_Znoise_2_ddpm_pred)
                sitk.WriteImage(synthetic_Znoise_2_ddpm_pred, os.path.join(save_dir, f"{pred_img_name}_ddpm_2noised.nii.gz"))

                synthetic_Z_ddpm_pred = synthetic_Z_ddpm.astype(np.float32)
                synthetic_Z_ddpm_pred = sitk.GetImageFromArray(synthetic_Z_ddpm_pred)
                sitk.WriteImage(synthetic_Z_ddpm_pred, os.path.join(save_dir, f"{pred_img_name}_ddpm_z.nii.gz"))

                ct_tfm = ct.astype(np.float32)
                ct_tfm = sitk.GetImageFromArray(ct_tfm)
                sitk.WriteImage(ct_tfm, os.path.join(save_dir, f"{pred_img_name}_ct.nii.gz"))

