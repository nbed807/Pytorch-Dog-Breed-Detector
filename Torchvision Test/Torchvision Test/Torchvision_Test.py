print("hello")

from torchvision import models
import torch

# Import the torchvision models as pretrained
squeezenet = models.squeezenet1_0(pretrained = True)
alexnet = models.alexnet(pretrained = True)
inception = models.inception_v3(pretrained = True)
vgg = models.vgg16(pretrained = True)


# Define and set up transform
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    ])

# Import Pillow
from PIL import Image
img = Image.open("golden.jpg")

# Transform Image
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Model Inference
alexnet.eval()
out1 = alexnet(batch_t)
print(out1.shape)

squeezenet.eval()
out2 = squeezenet(batch_t)
print(out2.shape)

inception.eval()
out3 = inception(batch_t)
print(out3.shape)

vgg.eval()
out4 = vgg(batch_t)
print(out4.shape)

# Read Labels
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


_, index = torch.max(out1, 1)

percentage1 = torch.nn.functional.softmax(out1, dim = 1)[0] * 100

print(classes[index[0]], percentage1[index[0]].item())


_, index = torch.max(out2, 1)

percentage2 = torch.nn.functional.softmax(out2, dim = 1)[0] * 100

print(classes[index[0]], percentage2[index[0]].item())


_, index = torch.max(out3, 1)

percentage3 = torch.nn.functional.softmax(out3, dim = 1)[0] * 100

print(classes[index[0]], percentage3[index[0]].item())


_, index = torch.max(out4, 1)

percentage4 = torch.nn.functional.softmax(out4, dim = 1)[0] * 100

print(classes[index[0]], percentage4[index[0]].item())


# If the prediction isn't right, the user could see the top five results
txt = input("Would you like to see the other predictions? [y/n] ")

if txt in ['y', 'Y', 'yes', 'Yes', 'YES']:

    _, indices = torch.sort(out1, descending=True)
    list = [(classes[idx], percentage1[idx].item()) for idx in indices[0][:5]]

    for item, percent in list:
        print(item, percent)


