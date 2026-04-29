import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class CustomDataset(TensorDataset):
    def __init__(self, data, targets, transform=None):
        super().__init__(data, targets)
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self.transform:
            x = x.view(1, 28, 28)
            x = self.transform(x)
            x = x.view(-1)
        return x, y

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    X_train = train_data.drop('label', axis=1).values / 255.0
    y_train = train_data['label'].values
    X_test = test_data.values / 255.0
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomErasing(p=0.1)
    ])
    
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor, transform=transform)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    
    num_epochs = 25
    print("\nTraining model...")
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (preds == targets).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f} (Best: {best_val_acc:.4f})")
    
    model.load_state_dict(torch.load('best_model.pth'))
    
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor.to(device))
        _, train_preds = torch.max(train_outputs, 1)
        train_accuracy = (train_preds == y_train_tensor.to(device)).float().mean().item()
        print(f"\nTraining accuracy: {train_accuracy:.4f}")
    
    print("\nGenerating predictions...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        _, test_preds = torch.max(test_outputs, 1)
        test_preds_np = test_preds.cpu().numpy()
    
    submission = pd.DataFrame({
        'ImageId': np.arange(1, len(test_preds_np) + 1),
        'Label': test_preds_np
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")

if __name__ == '__main__':
    main()