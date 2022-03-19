#!/bin/bash 

cd hf

pip uninstall -y torch torchaudio torchtext torchvision
pip install -r requirements.txt -q

read -s -p "Upload hf.txt: " ignore
mkdir ~/.huggingface
mv hf.txt ~/.huggingface/token
git config --global credential.helper store

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

mkdir ~/.kaggle
read -s -p "Upload kaggle.json: " ignore
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

mkdir data
kaggle competitions download -c nbme-score-clinical-patient-notes -p data
unzip data/nbme-score-clinical-patient-notes.zip

TRANSFORMERS_PATH="$(python -c 'import transformers; from pathlib import Path; print(Path(transformers.__file__).parent)')"
cp convert_slow_tokenizer.py $TRANSFORMERS_PATH/
cp tokenization_deberta_v2_fast.py $TRANSFORMERS_PATH/models/deberta_v2/
cp tokenization_deberta_v2.py $TRANSFORMERS_PATH/models/deberta_v2/

read -s -p "Upload wandb.txt: " ignore
export WANDB_API_KEY=$( cat wandb.txt )
wandb login