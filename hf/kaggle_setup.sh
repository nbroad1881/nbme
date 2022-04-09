#!/bin/bash 

pip install git+https://github.com/huggingface/transformers.git -q
pip install -U wandb -q

TRANSFORMERS_PATH="$(python -c 'import transformers; from pathlib import Path; print(Path(transformers.__file__).parent)')"
cp /kaggle/input/nbmetoolkit/convert_slow_tokenizer.py $TRANSFORMERS_PATH/
cp /kaggle/input/nbmetoolkit/tokenization_deberta_v2_fast.py $TRANSFORMERS_PATH/models/deberta_v2/
cp /kaggle/input/nbmetoolkit/tokenization_deberta_v2.py $TRANSFORMERS_PATH/models/deberta_v2/