import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

def train_model(data_dir, num_epochs=20, batch_size=16, learning_rate=0.0001):
    # 1. Define transforms with Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir)
    print(f"Loaded {len(full_dataset)} images across classes: {full_dataset.classes}")

    # 3. Dynamic Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    # Apply specific transforms to each split
    class DatasetFromSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    # Use basic Subset for loading, then wrap for transforms
    from torch.utils.data import Subset
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_data = DatasetFromSubset(train_subset, transform=train_transform)
    val_data = DatasetFromSubset(val_subset, transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # 4. Initialize Model (ResNet18 weights updated)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Unfreeze some layers for better fine-tuning on small dataset
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))
    model = model.to(device)

    # 5. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 6. Training Loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': full_dataset.classes
            }, 'best_flower_model.pth')
            print(f"Model saved! (Acc: {val_acc:.2f}%)")

    print(f"Training finished. Best Val Acc: {best_acc:.2f}%")

    print(f"Training finished. Best Val Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    # Pointing to the current directory which contains daisy, rose, tulip folders
    # We need to ensure ImageFolder only sees those. List_dir showed:
    # .git, .nonadmin, LICENSE_PYTHON.txt, README.md, Untitled.ipynb, etc.
    # ImageFolder will fail if there are files in the root. 
    # Let's create a symbolic link style or just a 'data' folder for it.
    
    root = os.getcwd()
    data_path = os.path.join(root, "data_temp")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        # Link or move relevant folders
        import shutil
        for cls in ['daisy', 'rose', 'tulip']:
            src = os.path.join(root, cls)
            dst = os.path.join(data_path, cls)
            if os.path.exists(src) and not os.path.exists(dst):
                # Using copytree because symlinks on windows can be tricky without admin
                shutil.copytree(src, dst)

    train_model(data_path, num_epochs=5)
