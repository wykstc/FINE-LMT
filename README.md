# MMT

This is the official implementation for the paper "FINE-LMT: Fine-grained Feature Learning
for Multi-Modal Machine Translation".

# Structure of FINE-LMT
![image](modelstructure.PNG)

## Requirement
- Python  3.9.17
- Pytorch 1.13.1
- Fairseq 0.12.2

## File 
Because it is the first, fast version of our project, we hope you can put "FeatureAlignment.py" in "/fairseq/fairseq/models", "language_pair_dataset.py" in "/fairseq/fairseq/data", "orthogonalLoss.py" in "/fairseq/fairseq/modules" and "multimodal_translation.py" in "/fairseq/fairseq/tasks".

## Data
- The processed text data is provided in the data-bin file.
- Image feature can be obtained by using "imageFeatureExtraction.py". The original image data can be download at [Flickr](https://www.statmt.org/wmt17/multimodal-task.html) and [Multi30k](https://github.com/multi30k/dataset)

### Train the model
```
bash train.sh
```

### Evaluate the model
When evaluate the model, please use the correponding image index in the "language_pair_dataset.py" file.
```
bash test.sh
```
