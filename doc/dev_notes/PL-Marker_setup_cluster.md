# setup

* `git clone https://github.com/thunlp/PL-Marker.git`
* `$ module load devel/cuda/11.0`
* `$ module load devel/python/3.8.6_intel_19.1`
* venv setup
    * `$ python -m venv venv`
    * `$ source venv/bin/activate`
    * `$ pip install --upgrade pip`
    * `$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
    * `$ pip install -r PL-Marker/requirement.txt`
    * `$ pip install --editable ./PL-Marker/transformers`
    * `$ pip install wheel`
    * `$ cd PL-Marker`
    * `$ git clone https://github.com/NVIDIA/apex`
    * `$ cd apex`
    * `$ pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./`
    * `$ cd ..`
    * `$ pip install gdown`
* data setup
    * `$ gdown --folder https://drive.google.com/drive/folders/1_ccNEm9LlqegoGXl69PJEbSW16Qvx7X7` (sciner-scibert)
    * `$ gdown --folder https://drive.google.com/drive/folders/1V0l3pdZVnvUcQXbjQgL5uJKivTlyT986`   (scire-scibert)
    * `mkdir -p bert_models/scibert_scivocab_uncased`
    * `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/pytorch_model.bin`
    * `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/vocab.txt`
    * `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/config.json`
* hyperpie dataset compatibility adjustments
    * `$ scp ys8950@aifb-ls3-icarus.aifb.kit.edu:/home/ws/ys8950/dev/PL-Marker/run_re.py .`
    * `$ scp ys8950@aifb-ls3-icarus.aifb.kit.edu:/home/ws/ys8950/dev/PL-Marker/run_acener.py .`

# test

* `$ salloc -p dev_gpu_4 --ntasks=1 --time=0:30:00 --gres=gpu:1 --mail-type=BEGIN,END,FAIL --mail-user=tarek.saier@kit.edu`
* `$ module load devel/cuda/11.0`
* `$ module load devel/python/3.8.6_intel_19.1`
* `$ cd pl-marker`
* `$ source /home/kit/aifb/ys8950/pl_marker/venv/bin/activate`

```
$ python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./hyperpie --learning_rate 2e-5 --num_train_epochs 5 --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file all_444.jsonl --dev_file all_444.jsonl --test_file trpp_15k_001 --output_dir ./sciner-hyperpie_predict_pprs_15_sample_001 --output_results
```

fails w/

```
AttributeError: module 'torch.distributed' has no attribute '_all_gather_base'
```
