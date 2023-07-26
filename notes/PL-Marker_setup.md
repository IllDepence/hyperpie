https://github.com/thunlp/PL-Marker

basic setup

* `git clone https://github.com/thunlp/PL-Marker.git`
* `cd PL-Marker`
* python3.8 venv setup
* pip upgrade
* `pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`
* `pip install -r requirement.txt`
* `pip install --editable ./transformers`

data

* `mkdir -p bert_models/scibert_scivocab_uncased`
* `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/pytorch_model.bin`
* `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/vocab.txt`
* `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/config.json`

create directory for model output and run quickstart script

* `mkdir -p sciner_models/sciner-scibert`
* `CUDA_VISIBLE_DEVICES=0  python3  run_acener.py  --model_type bertspanmarker      --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case      --data_dir data/scierc     --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1      --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8        --do_eval  --evaluate_during_training   --eval_all_checkpoints      --fp16  --seed 42  --onedropout  --lminit      --train_file train.json --dev_file dev.json --test_file test.json      --output_dir sciner_models/sciner-scibert  --overwrite_output_dir  --output_results`

runs but does not seem to work properly

```
07/26/2023 11:06:41 - INFO - __main__ -   Evaluate on test set
07/26/2023 11:06:41 - INFO - __main__ -   Evaluate the following checkpoints: []
07/26/2023 11:06:41 - INFO - __main__ -   Result: {"dev_best_f1": 0}
```

install apex although it’s deprecated

* `git clone https://github.com/NVIDIA/apex`
* `cd apex`
* `pip install wheel`
* `pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./`

run training

```
CUDA_VISIBLE_DEVICES=0  python3  run_acener.py  --model_type bertspanmarker  \
    --model_name_or_path bert_models/scibert_scivocab_uncased --do_lower_case  \
    --data_dir scierc  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir sciner_models/PL-Marker-scierc-scibert-$seed  --overwrite_output_dir  --output_results
```

note: get `Warning:  multi_tensor_applier fused unscale kernel is unavailable, possibly because apex was installed without --cuda_ext --cpp_ext. Using Python fallback.  Original ImportError was: ModuleNotFoundError("No module named 'amp_C'")`

but runs ­ after 20/50 epochs:
```
07/26/2023 13:31:08 - INFO - __main__ -   Result: {"f1": 0.7208307880268784, "f1_overlap": 0.722323049001815, "precision": 0.7142857142857143, "recall": 0.7274969173859432}5<00:00,  3.89it/s]
```
