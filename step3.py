# clearml_requirements: torch==2.0.1+cpu, torchvision==0.15.2+cpu, timm, numpy, matplotlib, clearml
# --extra-index-url https://download.pytorch.org/whl/cpu

from clearml import Task, Dataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ✅ ClearML Task Init
task = Task.init(
    project_name="plantdataset",
    task_name="Step 3 - Hybrid Model Training (AIS-Personal)"
)

# ✅ Load dataset from ClearML
dataset = Dataset.get(dataset_name="New Augmented Plant Disease Dataset")
dataset_path = dataset.get_local_copy()

# ✅ Paths
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")

# ✅ Transform (same as AIS-Personal)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# ✅ Load Pretrained Models
device = torch.device("cpu")

mobilenet = models.mobilenet_v2(pretrained=True).features.to(device)
densenet = models.densenet121(pretrained=True).features.to(device)

mobilenet.eval()
densenet.eval()

# ✅ Hybrid Classifier Head
class HybridNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridNet, self).__init__()
        self.mobilenet = mobilenet
        self.densenet = densenet
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            x1 = self.mobilenet(x)
            x1 = self.pool(x1)
            x1 = self.flatten(x1)

            x2 = self.densenet(x)
            x2 = self.pool(x2)
            x2 = self.flatten(x2)

        x_cat = torch.cat((x1, x2), dim=1)
        return self.classifier(x_cat)

model = HybridNet(num_classes=len(train_data.classes)).to(device)

# ✅ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ✅ Train
epochs = 2  # (use more in real run)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()

    accuracy = correct / len(train_data)
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={accuracy:.4f}")

print("✅ Training Complete")
