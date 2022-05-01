#!/bin/bash 

cd nbme/hf

# pip uninstall -y torch torchaudio torchtext torchvision
# pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt

FILE="hf.txt"
if test -f "~/.huggingface/token"; then
    echo "$FILE exists."
else
    read -s -p "Upload hf.txt: " ignore
    mkdir ~/.huggingface
    mv hf.txt ~/.huggingface/token
fi

git config --global credential.helper store

apt-get install git-lfs
git lfs install

FILE="~/.kaggle/kaggle.json"
if test -f "~/.kaggle/kaggle.json"; then
    echo "$FILE exists."
else
    mkdir ~/.kaggle
    read -s -p "Upload kaggle.json: " ignore
    mv kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

    mkdir data
    kaggle competitions download -c nbme-score-clinical-patient-notes -p data
    unzip data/nbme-score-clinical-patient-notes.zip -d data
fi

FILE="wandb.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
    read -s -p "Upload wandb.txt: " ignore
fi

export WANDB_API_KEY=$( cat wandb.txt )
wandb login

export GIT_EMAIL=$( cat email.txt )
export GIT_NAME=$( cat name.txt )

git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"