# MMT

This is the official implementation for the paper "EMP: Emotion-guided Multi-modal Fusion and Contrastive
Learning for Personality Traits Recognition".

# Structure of EMP
![image](modelstructure.png)

## Requirement
Python  3.9.17
Pytorch 1.13.1
Fairseq 0.12.2

# File 
Because it is the first, fast version of our project, we hope you can put "FeatureAlignment.py" in "/fairseq/fairseq/models", "language_pair_dataset.py" in "/fairseq/fairseq/data", "orthogonalLoss.py" in "/fairseq/fairseq/modules" and "multimodal_translation.py" in "/fairseq/fairseq/tasks".

### Train the model
```
bash train.sh
```

### evaluate the model
When evaluate the model, please use the correponding image index in the "language_pair_dataset.py" file.
```
bash test.sh
```
