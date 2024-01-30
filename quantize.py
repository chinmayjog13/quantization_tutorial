import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

import warnings

warnings.filterwarnings('ignore')

model_path = 'flowers_model.pth'
quantized_model_save_path = 'quantized_flowers_model.pth'
batch_size = 10
num_classes = 102

# Define data transforms
transform = transforms.Compose(
                                [transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                            (0.485, 0.465, 0.406), 
                                            (0.229, 0.224, 0.225))]
)

# Define train data loader, for using as calibration set
trainset = torchvision.datasets.Flowers102(root='./data', split="train", 
                                           download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle=True, num_workers=2)

# Load the finetuned resnet model
model_to_quantize = resnet18(weights=None)
num_features = model_to_quantize.fc.in_features
model_to_quantize.fc = nn.Linear(num_features, num_classes)
model_to_quantize.load_state_dict(torch.load(model_path))
model_to_quantize.eval()
print('Loaded fine-tuned model')

# Define quantization parameters config for the correct platform, 
# "x86" for x86 devices or "qnnpack" for arm devices
qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)

# Fuse conv-> relu, conv -> bn -> relu layer blocks and insert observers
model_prep = prepare_fx(model=model_to_quantize, 
                        qconfig_mapping=qconfig_mapping, 
                        example_inputs=torch.randn((1,3,224,224)))

# Run calibration for 10 batches (100 random samples in total)
print('Running calibration')
with torch.no_grad():
    for i, data in enumerate(trainLoader):
        samples, labels = data

        _ = model_prep(samples)

        if i == 10: 
            break

# Quantize calibrated model
quantized_model = convert_fx(model_prep)
print('Quantized model!')

# Save quantized model
torch.save(quantized_model.state_dict(), quantized_model_save_path)
print('Saved quantized model weights to disk')

print('\nPrinting conv1 layer of fp32 and quantized model')
print(f'fp32 model: {model_to_quantize.conv1}')
print(f'quantized model: {quantized_model.conv1}')