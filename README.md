# DeepSTARR-Reimplementation
This is the individual final project for Statistical Program for Big Data (BIOSTAT823) 24 fall at Duke

## To create the conda environment for this re-implementation, run commands:
```
cd DeepSTARR-Reimplementation
conda env create -f requirements.yml -n re-deepstarr
conda activate re-deepstarr
```

## To train re-implemented DeepSTARR model, run commands:

### download dataset
```
wget -P data https://data.starklab.org/almeida/DeepSTARR/Tutorial/Sequences_activity_all.txt
```
### make your output directory
```
mkdir outputs 
python Deepstarr-train.py config/config-baseline.json data/Sequences_activity_all.txt outputs baseline
```

## To calculate nucleotide contribution scores, run commands after model training:
```
python nucleotide_contribution_score.py config/config-baseline.json data/Sequences_activity_all.txt outputs/baseline.h5 outputs/
```
