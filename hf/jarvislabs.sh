#!/bin/bash 

cd nbme/hf

# pip uninstall -y torch torchaudio torchtext torchvision
# pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt

read -s -p "Upload hf.txt: " ignore
mkdir ~/.huggingface
mv hf.txt ~/.huggingface/token
git config --global credential.helper store

apt-get install git-lfs
git lfs install

mkdir ~/.kaggle
read -s -p "Upload kaggle.json: " ignore
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

mkdir data
kaggle competitions download -c nbme-score-clinical-patient-notes -p data
unzip data/nbme-score-clinical-patient-notes.zip -d data

read -s -p "Upload wandb.txt: " ignore
export WANDB_API_KEY=$( cat wandb.txt )
wandb login