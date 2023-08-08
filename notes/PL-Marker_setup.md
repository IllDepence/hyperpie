# Basic setup

* `git clone https://github.com/thunlp/PL-Marker.git`
* `cd PL-Marker`
* python3.8 venv setup
* pip upgrade
* `pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`
* `pip install -r requirement.txt`
* `pip install --editable ./transformers`

## Data & Model prep

* `mkdir -p bert_models/scibert_scivocab_uncased`
* `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/pytorch_model.bin`
* `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/vocab.txt`
* `wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/config.json`

## Getting to run

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

### Test NER

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
final

```
07/26/2023 14:26:27 - INFO - __main__ -   ***** Running evaluation  *****
07/26/2023 14:26:27 - INFO - __main__ -     Num examples = 619
07/26/2023 14:26:27 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:10<00:00,  3.56it/s]
07/26/2023 14:26:38 - INFO - __main__ -     Evaluation done in total 10.983758 secs (56.355940 example per second)                                                                            
07/26/2023 14:26:38 - INFO - __main__ -   Result: {"f1": 0.6972697269726972, "f1_overlap": 0.6937926937926938, "precision": 0.7050970873786407, "recall": 0.6896142433234421}                 
07/26/2023 14:26:38 - INFO - __main__ -   Result: {"dev_best_f1": 0.7402049427365883, "f1_": 0.6972697269726972, "f1_overlap_": 0.6937926937926938, "precision_": 0.7050970873786407, "recall_": 0.6896142433234421}
```

## Full test run

### NER with reduced epoch size

```
$ CUDA_VISIBLE_DEVICES=0 python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir scierc --learning_rate 2e-5 --num_train_epochs 5 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512 --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout  --lminit --train_file train.json --dev_file dev.json --test_file test.json --output_dir ./bert_models/scibert_scivocab_uncased --overwrite_output_dir --output_results
```
**NOTE**: “model-path and the output-path have to be the same” ([source](https://github.com/thunlp/PL-Marker/issues/13#issuecomment-1036084244)), otherwise there seems to be no `ent_pred_test.json` output

```
$ ls bert_models/scibert_scivocab_uncased
checkpoint-1315    ent_pred_test.json  scripts                  training_args.bin
config.json        pytorch_model.bin   special_tokens_map.json  vocab.txt
ent_pred_dev.json  results.json        tokenizer_config.json
```

### RE fails

```
$ CUDA_VISIBLE_DEVICES=0  python3  run_re.py  --model_type bertsub      --model_name_or_path  ./bert_models/scibert_scivocab_uncased  --do_lower_case      --data_dir scierc      --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1      --max_seq_length 512  --max_pair_length 16  --save_steps 2500      --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax       --test_file ent_pred_test.json      --use_ner_results     --output_dir ./bert_models/scibert_scivocab_uncased
```

fails with

```
Traceback (most recent call last):
  File "run_re.py", line 1333, in <module>
    main()
  File "run_re.py", line 1312, in main
    model = model_class.from_pretrained(checkpoint, config=config)
  File "/home/ws/ys8950/dev/PL-Marker/transformers/src/transformers/modeling_utils.py", line 560, in from_pretrained
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for BertForACEBothOneDropoutSub:
        size mismatch for ner_classifier.weight: copying a param with shape torch.Size([7, 3072]) from checkpoint, the shape in current model is torch.Size([7, 1536]).
```

# Downloading more stuff

* `$ pip install gdown`
* `$ gdown --folder https://drive.google.com/drive/folders/1_ccNEm9LlqegoGXl69PJEbSW16Qvx7X7` (sciner-scibert)
* `$ gdown --folder https://drive.google.com/drive/folders/1V0l3pdZVnvUcQXbjQgL5uJKivTlyT986` (scire-scibert)

## Successful NER eval run

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir scierc --learning_rate 2e-5 --num_train_epochs 5 --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_eval  --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.json --dev_file dev.json --test_file test.json --output_dir ./sciner-scibert --overwrite_output_dir --output_results
```

```
08/08/2023 15:49:08 - INFO - __main__ -   Evaluate on test set
08/08/2023 15:49:08 - INFO - __main__ -   Evaluate the following checkpoints: ['./sciner-scibert']
08/08/2023 15:49:08 - INFO - transformers.modeling_utils -   loading weights file ./sciner-scibert/pytorch_model.bin
08/08/2023 15:49:11 - INFO - __main__ -   maxL: 313
08/08/2023 15:49:11 - INFO - __main__ -   maxR: 748
08/08/2023 15:49:11 - INFO - __main__ -   ***** Running evaluation  *****
08/08/2023 15:49:11 - INFO - __main__ -     Num examples = 619
08/08/2023 15:49:11 - INFO - __main__ -     Batch size = 16
Evaluating: 100%|██████████████████████████████████████████████| 39/39 [00:12<00:00,  3.09it/s]
08/08/2023 15:49:23 - INFO - __main__ -     Evaluation done in total 12.625185 secs (49.028984 example per second)
08/08/2023 15:49:23 - INFO - __main__ -   Result: {"f1": 0.7009456264775414, "f1_overlap": 0.6971027216856892, "precision": 0.6980576809888169, "recall": 0.7038575667655786}
08/08/2023 15:49:23 - INFO - __main__ -   Result: {"dev_best_f1": 0, "f1_": 0.7009456264775414, "f1_overlap_": 0.6971027216856892, "precision_": 0.6980576809888169, "recall_": 0.7038575667655786}
```

## Successful RE eval run

**copy NER result**: `cp sciner-scibert/ent_pred_test.json scierc/`

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./scierc --learning_rate 2e-5  --num_train_epochs 10 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16  --test_file ent_pred_test.json --use_ner_results --output_dir ./scire-scibert
```

```
08/08/2023 15:56:09 - INFO - __main__ -   Evaluate the following checkpoints: ['./scire-scibert']
08/08/2023 15:56:09 - INFO - transformers.modeling_utils -   loading weights file ./scire-scibert/pytorch_model.bin
08/08/2023 15:56:11 - INFO - __main__ -   maxR: 10
08/08/2023 15:56:11 - INFO - __main__ -   maxL: 338
08/08/2023 15:56:11 - INFO - __main__ -   ***** Running evaluation  *****
08/08/2023 15:56:11 - INFO - __main__ -     Batch size = 16
08/08/2023 15:56:11 - INFO - __main__ -     Num examples = 1699
Evaluating: 100%|████████████████████████████████████████████| 107/107 [00:05<00:00, 18.02it/s]
08/08/2023 15:56:17 - INFO - __main__ -     Evaluation done in total 5.965029 secs (92.371730 example per second)
08/08/2023 15:56:17 - INFO - __main__ -   Result: {"f1": 0.5390254420008624, "f1_with_ner": 0.4286330314790858, "ner_f1": 0.7009456264775414}
{'dev_best_f1': 0, 'f1_': 0.5390254420008624, 'f1_with_ner_': 0.4286330314790858, 'ner_f1_': 0.7009456264775414}
```

## Successful NER train+eval run

```
CUDA_VISIBLE_DEVICES=0  python3 run_acener.py --model_type bertspanmarker --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir scierc --learning_rate 2e-10 --num_train_epochs 1 --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16 --gradient_accumulation_steps 1 --max_seq_length 512  --save_steps 2000 --max_pair_length 256 --max_mention_ori_length 8 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --fp16 --seed 42 --onedropout --lminit --train_file train.json --dev_file dev.json --test_file test.json --output_dir ./sciner-scibert_bad --output_results
```

## Successful RE train+eval run

`cp sciner-scibert_bad/ent_pred_test.json scierc/`

```
CUDA_VISIBLE_DEVICES=0 python3 run_re.py --model_type bertsub --model_name_or_path ./bert_models/scibert_scivocab_uncased --do_lower_case --data_dir ./scierc --learning_rate 2e-15  --num_train_epochs 1 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1 --max_seq_length 256 --max_pair_length 16 --save_steps 2500 --do_train --do_eval --evaluate_during_training --eval_all_checkpoints --eval_logsoftmax --fp16  --test_file ent_pred_test.json --use_ner_results --output_dir ./scire-scibert_bad
```

# Data notes

[notes on data format and preprocessing](https://github.com/thunlp/PL-Marker/issues/11)

token replacements (seen in SciERC and found below mapping in ontonotes preprocessing script)

```
"-LRB-": "(",
"-RRB-": ")",
"-LSB-": "[",
"-RSB-": "]",
"-LCB-": "{",
"-RCB-": "}",
```

## Using with new data

Need to adjust number of labels ([source](https://github.com/thunlp/PL-Marker/issues/35#issuecomment-1289256473))

```
PL-Marker/run_acener.py

Lines 939 to 946 in 07fde08
 if args.data_dir.find('ace')!=-1: 
     num_labels = 8 
 elif args.data_dir.find('scierc')!=-1: 
     num_labels = 7 
 elif args.data_dir.find('ontonotes')!=-1: 
     num_labels = 19 
 else: 
     assert (False) 
```
