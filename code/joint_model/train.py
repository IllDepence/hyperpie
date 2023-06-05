# %%
import os
import json
import random
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import string
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import wandb

# self defined
from models.base_model import Encoder
from models.data import parse_file, Collate_Fn_Manager
from models.util import set_seed

# %%
# random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
# print(random_string)

current_datetime = datetime.now()
folder_name = current_datetime.strftime("%Y%m%d_%H:%M:%S")
# os.mkdir(folder_name)
print(folder_name)

parser = argparse.ArgumentParser()
# parser.add_argument("--seed_model", type=int, default=43, 
#                     help="random seed for model")
# parser.add_argument("--seed_data", type=int, default=43, 
#                     help="random seed for data")
parser.add_argument("--seed", type=int, default=42, 
                    help="random seed for data")
parser.add_argument("--num_epochs", type=int, default=1, 
                    help="number of epochs to train")
parser.add_argument("--batch_size_eval", type=int, default=1, 
                    help="eval batch size")
parser.add_argument("--batch_size_training", type=int, default=4, 
                    help="training batch size")
parser.add_argument("--warmup_epochs", type=int, default=1, 
                    help="warmup epochs")
parser.add_argument("--learning_rate", type=float, default=7e-5, 
                    help="learning rate")
parser.add_argument("--project", type=str, default="RE1st", 
                    help="project name for wandb")
parser.add_argument("--preprocessing", type=str, default=None, 
                    help="no preprocessing & latex2text preprocessing: None, latex2text.")
parser.add_argument("--model", type=str, default="base", 
                    help="model")
parser.add_argument("--language_model", type=str, default="allenai/scibert_scivocab_cased", 
                    help="language model")
parser.add_argument("--candidate_downsampling", type=int, default=1000, 
                    help="number of candidate spans to use during training (-1 for no downsampling)")
parser.add_argument("--negative_prob", type=float, default=1.0, 
                    help="probability of showing negative relation examples")
parser.add_argument("--k_mentions", type=int, default=50, 
                    help="number of mention spans to perform relation extraction on")
parser.add_argument("--k_mentions_test", type=int, default=400, 
                    help="number of mention spans to perform relation extraction on at test time")
parser.add_argument("--pooling", type=str, default="max", 
                    help="mention pooling method: max, mean")

args = parser.parse_args()

# set the wandb project where this run will be logged
# wandb.init(
#             project=args.project,
#             config = {
#                 'learning_rate': args.learning_rate,
#                 'epoch': args.num_epochs
#             }
# )
# wandb.config.update(args)
# wandb.config.identifier = folder_name

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
g = torch.Generator()
g.manual_seed(args.seed)
# set_seed(args.seed_model)
# torch.manual_seed(args.seed)

# -------------- data init --------------
tokenizer = AutoTokenizer.from_pretrained(args.language_model)

train_files = os.listdir('data/raw_training/')
parsed_files = []
for file in train_files:
    if file.endswith(".json"):
        output = parse_file(f"data/raw_training/{file}", tokenizer=tokenizer, preprocessing=args.preprocessing)
        parsed_files.extend(output)
        # parsed_files.extend(parse_file(f"data/raw_training/{file}", tokenizer=tokenizer, preprocessing=args.preprocessing))

train_loader = DataLoader(parsed_files, 
                          batch_size=args.batch_size_training, 
                          shuffle=True, 
                          collate_fn=Collate_Fn_Manager(max_span_len=15).collate_fn,
                          generator=g)


dev_files = os.listdir('data/raw_dev/')
parsed_files_dev = []
for file in dev_files:
    if file.endswith(".json"):
        output = parse_file(f"data/raw_dev/{file}", tokenizer=tokenizer, preprocessing=args.preprocessing)
        parsed_files_dev.extend(output)

dev_loader = DataLoader(parsed_files_dev, 
                        batch_size=args.batch_size_eval, 
                        shuffle=False, 
                        collate_fn=Collate_Fn_Manager(max_span_len=15).collate_fn)


test_files = os.listdir('data/raw_test/')
parsed_files_test = []
for file in test_files:
    if file.endswith(".json"):
        output = parse_file(f"data/raw_test/{file}", tokenizer=tokenizer, preprocessing=args.preprocessing)
        parsed_files_test.extend(output)

test_loader = DataLoader(parsed_files_test, 
                         batch_size=1, 
                         shuffle=False, 
                         collate_fn=Collate_Fn_Manager(max_span_len=15).collate_fn)

print('data loader finished ...')

# # -------------- model loading --------------
# start by loading language model

lm_config = AutoConfig.from_pretrained(args.language_model,
                                       num_labels=10)
lm_model = AutoModel.from_pretrained(args.language_model,
                                    from_tf=False,
                                    config=lm_config,)

# if args.model == "base":
#     from models.base_model import Encoder

model = Encoder(config=lm_config,
                model=lm_model, 
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                negative_prob=args.negative_prob,
                k_mentions=args.k_mentions,
                pooling=args.pooling)

model.to('cuda')

print('Model load finised ...')
# -------------- optimizer things --------------
optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6)
total_steps = len(train_loader) * args.num_epochs
warmup_steps = len(train_loader) * args.warmup_epochs
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

scaler = GradScaler()
# %% --------------- TRAIN LOOP ------------------

best_dev_f1 = 0
step_global = 0

for ep in tqdm(range(args.num_epochs)):
    avg_rloss = []
    avg_mloss = []
    avg_loss = []
    model.true_positives = 0
    model.false_positives = 0
    model.false_negatives = 0
    model.true_negatives = 0
    
    model.train()
    with tqdm(train_loader) as progress_bar:
        for b in progress_bar:
            step_global += 1
            input_ids, attention_masks, candidate_spans, relation_labels, offset_mapping, ids, sources = b

            if input_ids.size(1) > 4096:
                # skip an insanely long document...
                continue

            # reduce candidate spans via random sampling
            if args.candidate_downsampling != -1:
                candidate_spans = [random.sample(x, min(args.candidate_downsampling, len(x))) for x in candidate_spans]
            # add back spans for which we have labels
            for i, b_i in enumerate(relation_labels):
                for label in b_i:
                    h_ent = label['h']
                    t_ent = label['t']
                    if h_ent not in candidate_spans[i]:
                        candidate_spans[i].append(h_ent)
                    if t_ent not in candidate_spans[i]:
                        candidate_spans[i].append(t_ent)
            
            # pass data to model
            with autocast():
                relation_loss, mention_loss, loss, output = model(input_ids.to('cuda'), attention_masks.to('cuda'), candidate_spans, relation_labels)

            # accumulate loss
            avg_rloss.append(relation_loss.item())
            avg_mloss.append(mention_loss.item())
            avg_loss.append(loss.item())
            
            # backpropagate & reset
            scaler.scale(loss).backward()
            
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()
            lr_scheduler.step()
            model.zero_grad()
            del loss

            acc, p, r, f1 = 0, 0, 0, 0
            if model.true_positives != 0:
                p = model.true_positives/(model.true_positives+model.false_positives)
                r = model.true_positives/(model.true_positives+model.false_negatives)
                f1 = (2*p*r)/(p+r)
            acc = (model.true_positives + model.true_negatives)/(model.true_positives + model.true_negatives + model.false_negatives + model.false_positives)
            progress_bar.set_postfix({"L":f"{sum(avg_loss)/len(avg_loss):.2f}", "TP":f"{model.true_positives}", "A":f"{100*acc:.2f}", "P":f"{100*p:.2f}", "R":f"{100*r:.2f}", "F1":f"{100*f1:.2f}"})

    #         wandb.log({"loss": avg_loss[-1]}, step=step_global)
    #         wandb.log({"loss_mention": avg_mloss[-1]}, step=step_global)
    #         wandb.log({"loss_relation": avg_rloss[-1]}, step=step_global)

    # wandb.log({"precision_train": p}, step=step_global)
    # wandb.log({"recall_train": r}, step=step_global)
    # wandb.log({"f1_micro_train": f1}, step=step_global)
    # wandb.log({"accuracy_train": acc}, step=step_global)

    # --------------- EVAL ON DEV ------------------
    avg_loss = []
    model.true_positives = 0
    model.false_positives = 0
    model.false_negatives = 0
    model.true_negatives = 0
    model.eval()
    with tqdm(dev_loader) as progress_bar:
        for b in progress_bar:

            input_ids, attention_masks, candidate_spans, relation_labels, offset_mapping, ids, sources = b

            if input_ids.size(1) > 4096:
                # skip an insanely long document...
                continue
            
            # pass data to model
            relation_loss, mention_loss, loss, output = model(input_ids.to('cuda'), attention_masks.to('cuda'), candidate_spans, relation_labels)

            # accumulate loss
            avg_loss.append(loss.item())
            del loss

            acc, p, r, f1 = 0, 0, 0, 0
            if model.true_positives != 0:
                p = model.true_positives/(model.true_positives+model.false_positives)
                r = model.true_positives/(model.true_positives+model.false_negatives)
                f1 = (2*p*r)/(p+r)
            acc = (model.true_positives + model.true_negatives)/(model.true_positives + model.true_negatives + model.false_negatives + model.false_positives)
            progress_bar.set_postfix({"L":f"{sum(avg_loss)/len(avg_loss):.2f}", "TP":f"{model.true_positives}", "A":f"{100*acc:.2f}", "P":f"{100*p:.2f}", "R":f"{100*r:.2f}", "F1":f"{100*f1:.2f}"})

        # if f1 > best_dev_f1:
        #     best_dev_f1 = f1
        #     wandb.log({"best_precision_dev": p}, step=step_global)
        #     wandb.log({"best_recall_dev": r}, step=step_global)
        #     wandb.log({"Best_f1_micro_dev": f1}, step=step_global)
        #     wandb.log({"Best_accuracy_dev": acc}, step=step_global)
        #     torch.save(model.state_dict(), f"checkpoints/{args.project}_{folder_name}.pt")

        # wandb.log({"precision_dev": p}, step=step_global)
        # wandb.log({"recall_dev": r}, step=step_global)
        # wandb.log({"f1_micro_dev": f1}, step=step_global)
        # wandb.log({"accuracy_dev": acc}, step=step_global)

# %%

print("---- TEST EVAL -----")
# --------------- EVAL ON TEST ------------------

if args.num_epochs > 0:
    model.load_state_dict(torch.load(f"checkpoints/{args.project}_{folder_name}.pt"))
else:
    step_global = 0

model.eval()

model.k_mentions = args.k_mentions_test

outputs = {}

with tqdm(test_loader) as progress_bar:
    for b in progress_bar:

        input_ids, attention_masks, candidate_spans, relation_labels, offset_mapping, ids, sources = b

        # pass data to model
        _, _, _, output = model(input_ids.to('cuda'), attention_masks.to('cuda'), candidate_spans)

        # convert label indexing from tokens to chars and save in correct format
        for preds, offsets, doc_id, source in zip(output, offset_mapping, ids, sources):
            detected_entities = []
            detected_relations = {}
            for prediction in preds:
                (h_span, h_type), (t_span, t_type), r_type = prediction
                h_span = [offsets[tk] for tk in h_span]
                h_ent = {
                    "label": h_type,
                    "start": h_span[0][0],
                    "end": h_span[1][1],
                }
                t_span = [offsets[tk] for tk in t_span]
                t_ent = {
                    "label": t_type,
                    "start": t_span[0][0],
                    "end": t_span[1][1],
                }
                
                if h_ent in detected_entities:
                    h_id = detected_entities.index(h_ent)
                else:
                    h_id = len(detected_entities)
                    detected_entities.append(h_ent)    

                if t_ent in detected_entities:
                    t_id = detected_entities.index(t_ent)
                else:
                    t_id = len(detected_entities)
                    detected_entities.append(t_ent)

                detected_relations[f"R{len(detected_relations)+1}"] = {
                    "rid": f"R{len(detected_relations)+1}",
                    "label": r_type,
                    "arg0": f"T{h_id+1}",
                    "arg1": f"T{t_id+1}",
                }
            
            ent_dict = {}
            for i, ent in enumerate(detected_entities):
                ent_dict[f"T{i+1}"] = {
                    "eid": f"T{i+1}",
                    "label": ent["label"],
                    "start": ent["start"],
                    "end": ent["end"],
                }

            doc = source
            doc["entity"] = ent_dict
            doc["relation"] = detected_relations
            file_id = doc["file"]
            del doc["file"]
            if file_id not in outputs.keys():
                outputs[file_id] = {}
            outputs[file_id][doc_id] = doc

        del output


os.makedirs(f'outputs/{args.project}_{folder_name}_test/')
tozip = []
for filename in outputs.keys():
    tozip.append(f'outputs/{args.project}_{folder_name}_test/{filename}')
    with open(f'outputs/{args.project}_{folder_name}_test/{filename}', 'w') as f:
        json.dump(outputs[filename], f, indent=4)

import zipfile
ZipFile = zipfile.ZipFile(f'outputs/{args.project}_{folder_name}_test/{args.project}_{folder_name}_test.zip', "w")
for a in tozip:
    ZipFile.write(a, os.path.basename(a), compress_type=zipfile.ZIP_DEFLATED)
ZipFile.close()
# artifact = wandb.Artifact(f'{folder_name}', 'predictions')
# artifact.add_file(f'outputs/{args.project}_{folder_name}_test/{args.project}_{folder_name}_test.zip')

# wandb.log_artifact(artifact)
