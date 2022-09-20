import torch
from torchvision import transforms

import pandas as pd
from PIL import Image
import requests
import os
import matplotlib.pyplot as plt
import natsort
import math

image_transforms = { 
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

index_to_class = {0: 'backpack', 1: 'bed', 2: 'chair', 3: 'couch', 4: 'laptop', 5: 'table'}
print (index_to_class)

def makePrediction(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = image_transforms['test']

    # test_image = Image.open(requests.get(url, stream=True).raw)
    test_image = Image.open(image_path)


    plt.imshow(test_image)
    
    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224).to(device)
    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        ps = torch.exp(out)
        
        topk, topclass = ps.topk(6, dim=1)
        print(topk)

        predictions = torch.argmax(ps)
        print(predictions)

        simple_least_conf_1 = topk[0][0]
        simple_least_conf_2 = topk[0][1]
        num_labels = topk.numel()

        #least confidence
        normalized_least_conf = (1 - simple_least_conf_1) * (num_labels / (num_labels -1))

        #margin confidence
        difference = simple_least_conf_1 - simple_least_conf_2
        margin_confidence = 1 - difference

        #entropy
        log_probs = topk * torch.log2(topk)
        raw_entropy = 0 - torch.sum(log_probs) 

        normalized_entropy = raw_entropy / math.log2(topk.numel())

        return predictions.item(), normalized_least_conf.item(), margin_confidence.item(), normalized_entropy.item()

        # for i in range(6):
        #     # print(f"Prediction {i+1} : {index_to_class[topclass.cpu().numpy()[0][i]]}, Score: {topk.cpu().numpy()[0][i] * 100}%")
        #     print(f"Prediction {i+1} : {index_to_class[topclass.cpu().numpy()[0][i]]}, Score: {topk.cpu().numpy()[0][i] * 100}%")

    
model = "model_res50.pt"
model = torch.load(model, map_location=torch.device('cuda:0'))

file_name = []
results_0 = []
results_1 = []
results_2 = []
results_3 = []
i = 0
image_path = '/home/fadilrisdian/project/task5/domain/day8/'
for filename in natsort.natsorted(os.listdir(image_path)):
    file = image_path + filename
    print(file)
    x = makePrediction(model, file)
    
    file_name.append(filename)
    results_0.append(x[0])
    results_1.append(x[1])
    results_2.append(x[2])
    results_3.append(x[3])

    print(x[0])
    print(i)
    i += 1

df = pd.DataFrame(results_0)
df.to_csv('day8_predict.csv',index=False)

