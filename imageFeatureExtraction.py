import pickle
import csv
import torch
from PIL import Image
import torchvision.models as models
from torchvision import transforms

resnet101 = models.resnet101(weights='IMAGENET1K_V1').to('cuda')

model = torch.nn.Sequential(*list(resnet101.children())[:-2]).to('cuda')



processor = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


imageFeature = {}
model.eval()


def getTrainData():
    train_setsID = []
    with open('data-bin/train.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            train_setsID.append(row[0])
    return train_setsID

def getValData():
    val_setsID = []
    with open('data-bin/val.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            val_setsID.append(row[0])
    return val_setsID

def get2016Data():
    val_setsID = []
    with open('data-bin/test_2016_flickr.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            val_setsID.append(row[0])
    return val_setsID

def get2017Data():
    val_setsID = []
    with open('data-bin/test_2017_flickr.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            val_setsID.append(row[0])
    return val_setsID

def getCoCoData():
    val_setsID = []
    with open('data-bin/test_2017_mscoco.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            val_setsID.append(row[0].split(".")[0] + '.jpg')
    return val_setsID


def featureExtraction(address,list):
    count = 0
    with torch.no_grad():
        for i in list:
            count = count + 1
            image = Image.open(address + i).convert("RGB")
            inputs = processor(image).to('cuda:0')
            outputs = model(inputs.unsqueeze(0)).view(1, 2048, -1).permute(0, 2, 1)
            imageFeature[i] = outputs.squeeze()
            if count%100==0:
                print(count)

print('train')
featureExtraction('MMT/flickr30k/flickr30k-images/',getTrainData())
print('val')
featureExtraction('MMT/flickr30k/flickr30k-images/',getValData())
print('2016')
featureExtraction('MMT/flickr30k/flickr30k-images/',get2016Data())
print('2017')
featureExtraction('MMT/flickr30k/test2017-images/',get2017Data())
print('coco')
featureExtraction('MMT/flickr30k/testcoco-images/',getCoCoData())

ff = open('Resnet101.pkl', 'wb')
pickle.dump(imageFeature, ff)
ff.close()

