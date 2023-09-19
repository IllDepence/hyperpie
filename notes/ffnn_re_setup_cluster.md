# Setup

## env

* `$ module load devel/python/3.10.0_intel_19.1`
* `$ module load devel/cuda/11.8`
* `$ python -m venv venv`
* `$ python -m pip install --upgrade pip`
* `$ python -m pip install tqdm numpy scikit-learn`
* `$ python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`
* `$ python -m pip install transformers`

## data

* `$ cd /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample`
* `$ cat sciner-hyperpie_predict_pprs_15_sample_*/ent_pred_test.json >> ent_pred_test_000-041.json`

# use

## manual

* `$ module load devel/python/3.10.0_intel_19.1`
* `$ module load devel/cuda/11.8`
* `$ cd /home/kit/aifb/ys8950/hyperpie_ffnn_re`
* `$ source /home/kit/aifb/ys8950/hyperpie_ffnn_re/venv/bin/activate`
* `$ python ffnn_re.py /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/train_dev_test.jsonl /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/ent_pred_test_000-041.json /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/ffnn_re_results_000-041.json`


# batch job

* `$ sbatch -p gpu_8 hyperpie_ffnn_re/ffnn_re_pred.sh`

```
#!/bin/bash
#SBATCH --job-name=ffnn_re_pred
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=ffnn_re_pred__%j.log
#SBATCH --mail-user=tarek.saier@kit.edu
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=256gb

pwd; hostname; date

module load devel/python/3.10.0_intel_19.1
module load devel/cuda/11.8

cd /home/kit/aifb/ys8950/hyperpie_ffnn_re
source /home/kit/aifb/ys8950/hyperpie_ffnn_re/venv/bin/activate

python ffnn_re.py /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/train_dev_test.jsonl /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/ent_pred_test_000-041.json /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/ffnn_re_results_000-041.json
```

fails w/ out of memory error → use smaller chunk of data

* `$ cat sciner-hyperpie_predict_pprs_15_sample_00*/ent_pred_test.json >> ent_pred_test_000-009.json`

```
[...]
#SBATCH --mem=500gb
[...]
python ffnn_re.py /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/train_dev_test.jsonl /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/ent_pred_test_000-009.json /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/ffnn_re_results_000-009.jso
```

fails w/ out of memory error

* opt to go with parallelism and merging results afterwards
* test single chunk on small compute node `$ salloc -p dev_single --ntasks=1 --mem=50000mb --time=0:30:00`

```
python ffnn_re.py /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/train_dev_test.jsonl /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/sciner-hyperpie_predict_pprs_15_sample_000/ent_pred_test.json /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/sciner-hyperpie_predict_pprs_15_sample_000/ffnn_re_results.json
```
runs through

```
Iteration 78, loss = 0.01847402
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
evaluating model...
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     47086
           1       0.19      0.47      0.27       438

    accuracy                           0.98     47524
   macro avg       0.59      0.73      0.63     47524
weighted avg       0.99      0.98      0.98     47524
```

* adjust script to take chunk parameter

```
$ cat hyperpie_ffnn_re/ffnn_re_pred_chunk.sh
#!/bin/bash
#SBATCH --job-name=ffnn_re_pred_XXX
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=ffnn_re_pred_XXX_%j.log
#SBATCH --mail-user=tarek.saier@kit.edu
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=100gb
#SBATCH --partition=single

pwd; hostname; date

module load devel/python/3.10.0_intel_19.1
module load devel/cuda/11.8

cd /home/kit/aifb/ys8950/hyperpie_ffnn_re
source /home/kit/aifb/ys8950/hyperpie_ffnn_re/venv/bin/activate

python ffnn_re.py /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/train_dev_test.jsonl /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/sciner-hyperpie_predict_pprs_15_sample_XXX/ent_pred_test.json /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/sciner-hyperpie_predict_pprs_15_sample_XXX/ffnn_re_results.json XXX
```

```
$ cat hyperpie_ffnn_re_run_batches.sh
#!/bin/bash
for yyy in `ls /pfs/work7/workspace/scratch/ys8950-general/hyperpie_PL-Marker_ER_15ksample/ | grep -Po 'sample_\d\d\d' | grep -Po '\d\d\d' | sort | uniq`
do
    cp hyperpie_ffnn_re/ffnn_re_pred_chunk.sh ffnn_re_pred_chunk_${yyy}.sh
    sed -i -E "s/XXX/${yyy}/g" ffnn_re_pred_chunk_${yyy}.sh
    # sbatch ffnn_re_pred_chunk_${yyy}.sh
    # sleep 0.2
    # rm ffnn_re_pred_chunk_${yyy}.sh
done
```
