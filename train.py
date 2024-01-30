import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from inference_utils import test_accuracy

# Define data transforms
transform = transforms.Compose(
                                [transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.465, 0.406), (0.229, 0.224, 0.225))]
)

batch_size = 4
num_epochs = 10
num_classes = 102
model_save_path = 'flowers_model.pth'

# Define train and test data loaders
trainset = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.Flowers102(root='./data', split="test", download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load pretrained model and modify last layer
model = resnet18(weights='ResNet18_Weights.DEFAULT')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model.train()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(trainLoader, 0):
        samples, labels = data

        optimizer.zero_grad()

        outputs = model(samples)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    print(f'Epoch {epoch+1}, loss: {running_loss / 100:.3f}')
    running_loss = 0.0

# Save trained model
print('Training done, saving model weights')
torch.save(model.state_dict(), model_save_path)

# Evaluate on test data
accuracy = test_accuracy(model, testLoader)
print(f'Accuracy on test set: {accuracy:.3f}')