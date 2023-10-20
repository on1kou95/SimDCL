# SimDCL
This is the official implementation in PyTorch for A Simple Framework for Depth-augmented Contrastive Learning.

<!-- 
The code is changed from https://github.com/kekmodel/FixMatch-pytorch -->

## Build with conda
Download data from anonymous account https://drive.google.com/drive/folders/1oXim4BxsQTq0U-fKZpX_LrL6J377WKZz?usp=share_link and unzip data.7z
```
conda env create -f SimDCL.yml
conda activate SimDCL
```
## Train

```
# to train the model by SimDCL:
python3 train.py --cfg ./configs/config_SimDCL.py

# to train the model by CCSSL:
python3 train.py --cfg ./configs/config_CCSSL.py
