import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
from inference_utils import test_accuracy, test_speed, load_quantized_model
import copy

import warnings
warnings.filterwarnings('ignore')

model_weights_path = 'flowers_model.pth'
quantized_model_weights_path = 'quantized_flowers_model.pth'
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

testset = torchvision.datasets.Flowers102(root='./data', split="test", 
                                          download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                         shuffle=False, num_workers=2)

# Load the finetuned resnet model and the quantized model
model = resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load(model_weights_path))
model.eval()

model_to_quantize = copy.deepcopy(model)
quantized_model = load_quantized_model(model_to_quantize, 
                                       quantized_model_weights_path)

# Compare accuracy
fp32_accuracy = test_accuracy(model, testLoader)
accuracy = test_accuracy(quantized_model, testLoader)

print(f'Original model accuracy: {fp32_accuracy:.3f}')
print(f'Quantized model accuracy: {accuracy:.3f}\n')

# Compare speed
fp32_speed = test_speed(model)
quantized_speed = test_speed(quantized_model)
print(f'Inference time for original model: {fp32_speed:.3f} ms')
print(f'Inference time for quantized model: {quantized_speed:.3f} ms\n')

# Compare file size
fp32_size = os.path.getsize(model_weights_path)/10**6
quantized_size = os.path.getsize(quantized_model_weights_path)/10**6
print(f'Original model file size: {fp32_size:.3f} MB')
print(f'Quantized model file size: {quantized_size:.3f} MB')