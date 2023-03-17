# SimDCL
This is the official implementation in PyTorch for A Simple Framework for Depth Estimation-based Contrastive Learning.

<!-- 
The code is changed from https://github.com/kekmodel/FixMatch-pytorch -->

## Build with conda

```
conda env create -f SimDCL.yml
conda activate SimDCL
```
## Train
down data from anonymous account
https://drive.google.com/drive/folders/1oXim4BxsQTq0U-fKZpX_LrL6J377WKZz?usp=share_link
and unzip data.7z
```
# to train the model by SimDCL:
python3 train.py --cfg ./configs/config_SimDCL.py --out result   --seed 5 --gpu-id 0

# to train the model by CCSSL:
python3 train.py --cfg ./configs/config_CCSSL.py --out result   --seed 5 --gpu-id 0
