# cnn_model_fixed.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# --------------------------
# 1. Device setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# 2. Transformations
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --------------------------
# 3. Load MNIST dataset
# --------------------------
train_dataset = datasets.MNIST(root='.', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='.', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------------------------
# 4. CNN Model
# --------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # (1,28,28) -> (32,28,28)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # (32,28,28) -> (64,28,28)
        self.pool = nn.MaxPool2d(2, 2)               # halves HxW

        self.relu = nn.ReLU()

        # After two conv+pool layers, image size: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))  # 28x28 -> 14x14
        x = self.pool(x)                          # 14x14 -> 7x7
        x = x.view(x.size(0), -1)                # dynamic flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model
model = CNNModel().to(device)

# --------------------------
# 5. Loss & optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 6. Training
# --------------------------
epochs = 3
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# --------------------------
# 7. Evaluation
# --------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
