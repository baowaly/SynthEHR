Synthesizing Electronic Health Records Using Improved Generative Adversarial Networks
================
Mrinal Kanti Baowaly, Chia-Ching Lin, Chao-Lin Liu, Kuan-Ta Chen

Publication:
------------

Mrinal Kanti Baowaly, Chia-Ching Lin, Chao-Lin Liu, Kuan-Ta Chen, Synthesizing electronic health records using improved generative adversarial networks, Journal of the American Medical Informatics Association, Volume 26, Issue 3, March 2019, Pages 228–241, https://doi.org/10.1093/jamia/ocy142

Our goal:
---------

The goal of this research is to generate synthetic electronic health records (EHRs) using two improved Generative Adversarial Networks: Wasserstein GAN with gradient penalty (WGAN-GP) and Boundary-seeking GAN (BGAN ). We defined the two models as medWGAN and medBGAN respectively.

> The generated EHRs will be more realistic than the existing works (e.g. medGAN) and these data will be free of legal, security and privacy concerns

Source code explanation:
------------------------
`model.py` defines the `MEDGAN`, `MEDWGAN`, and `MEDBGAN` classes, which will be imported in `train.py` to build the neural network for GAN training and EHR generation.

How to generate EHRs:
---------------------

### Step 1: Installation
Install 'Tensorflow' and download/clone the source code `model.py` and `train.py`
### Step 2: Prepare the training data
Download the MIMIC-III data, aggregate the medical codes (e.g. diagonsis codes, medication codes, or procedure codes) for each patient, and save them as an numpy data file (.npy file)
### Step 3: Train the GAN and generate EHR
- Usage:  
  ```console
  $ python train.py --data_file [path to the training data (npy format)] --n_pretrain_epoch 100 --n_epoch 1000
  ```
-   During training, a progress bar will be showed for each epoch. Also, a folder will be created (`medGAN` by default) and two subfolders `model` and `output` will be created therein, with the former containing model checkpoints and the latter containing the synthetic EHR data (called `generated.npy` by default).
-   To specify output folder name, add parameter `--model [model_name]`, where [model_name] is `medGAN` with any prefix or postfix, such as `medGAN_n_epoch_500`.
-   To run improved GAN, add parameter `--model medWGAN` or `--model medBGAN`. Again, `medWGAN` and `medBGAN` can also have any prefix or postfix.
-   For more parameters, please refer to the source code in `train.py`.
