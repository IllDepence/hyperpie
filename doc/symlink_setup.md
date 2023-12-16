# Setup

* setup python3.7 venv
* pip upgrade
* `pip install PyYAML==5.3`
* `pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
* `pip install pytorch-lightning==0.9.0`
* `pip install pandas==1.3.3`
* `pip install scikit-learn`
* `pip install transformers==4.10.2`
* `pip install protobuf==3.20.*`

in `NER/scripts/mrc_ner/reproduce/semeval.sh`, adjust paths

```
REPO_PATH=/home/ws/ys8950/dev/symlink/NER
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
DATA_DIR=/home/ws/ys8950/dev/symlink/NER/data/semeval
BERT_DIR=/home/ws/ys8950/dev/symlink/
```

adjust GPU params

```
CUDA_VISIBLE_DEVICES=0,1 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--gpus="1" \
--workers 1 \
```

remove ddp parameters (probably not necessary, tested before realizing silent preprocessing takes up time at first run)

```
--distributed_backend=ddp \
```

# Training

todo: add notes for ~~changing of hard coded paths and GPU parameters~~, where to put data, ~~working training script call, data preprocessing time requirement, etc.~~

* `NER/datasets/prepro_ner.py`

* training run as `symlink$ bash ./NER/scripts/mrc_ner/reproduce/semeval.sh`



```
Epoch 101:  43%|███████████████████████████████████████████████████                                                                   | 2384/5515 [04:24<05:47,  9.01it/s, loss=0.001, v_num=0]
dataset getitem
training step
forward
computing loss
^C/home/ws/ys8950/dev/symlink/venv/lib/python3.7/site-packages/pytorch_lightning/utilities/distributed.py:37: UserWarning: Detected KeyboardInterrupt, attempting graceful shutdown...
  warnings.warn(\*args, \*\*kwargs)
Saving latest checkpoint..
Epoch 101:  43%|███████████████████████████████████████████████████                                                                   | 2384/5515 [04:24<05:47,  9.00it/s, loss=0.001, v\_num=0]
training done, computing metrics
```

from `NER/output/large_lr3e-5_drop0.1_norm1.0_weight0.1_warmup500_maxlen512/eval_result_log.txt`

```
Epoch 00100: span_f1  was not in top 3
2023-07-27 09:33:15,995 - lightning - Saving latest checkpoint..
2023-07-27 09:33:16,470 - __main__ - =&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&
2023-07-27 09:33:16,470 - __main__ - Best F1 on DEV is 0.63185
2023-07-27 09:33:16,470 - __main__ - Best checkpoint on DEV set is /home/ws/ys8950/dev/symlink/NER/output/large_lr3e-5_drop0.1_norm1.0_weight0.1_warmup500_maxlen512/epoch=82.ckpt
2023-07-27 09:33:17,078 - __main__ - =&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&=&
```


# Inference

in `bash ./NER/scripts/mrc_ner/nested_inference.sh` adjust paths to model checkpoint (in output dir of training) and hyperparams YAML (in output dir -> lightning)

* run as `bash ./NER/scripts/mrc_ner/nested_inference.sh`
