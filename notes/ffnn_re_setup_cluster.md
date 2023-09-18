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
