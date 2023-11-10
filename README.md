# CM3dLDM

The Cross Modality 3-Dimensional Latent Diffusion Model (CM3dLDM) introduces a novel image synthesis method for computed tomography (CT) volume generation from magnetic resonance imaging (MRI). 
The provided code may be adapted for the synthesis of any modality, given the proper dataset is provided during training.

# How to Use

Pull the main branch with: git pull https://github.com/AustinTapp/CM3dLDM.git

Ensure all packages are installed (requirements.txt).

Sign up for the SynthRad Challenge to obtain and use the data, found here: https://synthrad2023.grand-challenge.org/Data/.
Download the training dataset. Be sure to run the restructure.py script.

<ins>For Training:</ins>
Designate the path to your data: 'export DATA=/home/data/SynthRad/brain_train_restruct/images'
Run CM3dLDM.py   

Alternative (recommended):
You may run CM3dLDM_VAE.py to perform only VAE training with a smaller batch size.
Then, increase the batch size in CM3dLDM_UNet.py and perform only DDPM training. 
This greatly improves the training speed of the overall process as it allows for hardware specific configuration.

<ins>For Inference:</ins>
Run infer.py

Visualize the results!

