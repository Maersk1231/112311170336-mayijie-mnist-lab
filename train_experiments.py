import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def train_model(exp_name, optimizer_type, lr, batch_size, data_aug, early_stopping, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== {exp_name} ===")
    print(f"Device: {device}, Optimizer: {optimizer_type}, LR: {lr}, Batch: {batch_size}")
    
    train_data = pd.read_csv('digit-recognizer (1)/train.csv')
    test_data = pd.read_csv('digit-recognizer (1)/test.csv')
    
    X_train = train_data.drop('label', axis=1).values / 255.0
    y_train = train_data['label'].values
    X_test = test_data.values / 255.0
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    val_accs = []
    best_val_acc = 0
    early_stop_count = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for data, targets in train_loader:
            if data_aug and epoch > 0:
                data = apply_data_augmentation(data)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (preds == targets).sum().item()
        
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_count = 0
            torch.save(model.state_dict(), f'model_{exp_name}.pth')
        else:
            early_stop_count += 1
        
        if early_stopping and early_stop_count >= 5:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    model.load_state_dict(torch.load(f'model_{exp_name}.pth'))
    model.eval()
    
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        _, train_preds = torch.max(train_outputs, 1)
        train_acc = (train_preds == y_train_tensor).float().mean().item()
    
    return train_acc, best_val_acc, train_losses

def apply_data_augmentation(x):
    x = x.view(-1, 1, 28, 28)
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
    ])
    x = transform(x)
    x = x.view(-1, 784)
    return x

experiments = [
    {'name': 'Exp1', 'optimizer': 'SGD', 'lr': 0.01, 'batch': 64, 'aug': False, 'early_stop': False},
    {'name': 'Exp2', 'optimizer': 'Adam', 'lr': 0.001, 'batch': 64, 'aug': False, 'early_stop': False},
    {'name': 'Exp3', 'optimizer': 'Adam', 'lr': 0.001, 'batch': 128, 'aug': False, 'early_stop': True},
    {'name': 'Exp4', 'optimizer': 'Adam', 'lr': 0.001, 'batch': 64, 'aug': True, 'early_stop': True},
]

results = []
all_losses = {}

for exp in experiments:
    train_acc, val_acc, losses = train_model(
        exp['name'], exp['optimizer'], exp['lr'], exp['batch'], exp['aug'], exp['early_stop']
    )
    results.append({
        'name': exp['name'],
        'train_acc': train_acc,
        'val_acc': val_acc,
        'losses': losses
    })
    all_losses[exp['name']] = losses
    print(f"Final: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

print("\n=== 实验结果汇总 ===")
for r in results:
    print(f"{r['name']}: Train={r['train_acc']:.4f}, Val={r['val_acc']:.4f}")

plt.figure(figsize=(10, 6))
for exp_name, losses in all_losses.items():
    plt.plot(range(1, len(losses)+1), losses, label=exp_name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')
print("\nLoss曲线已保存为 loss_curves.png")

pd.DataFrame(results).to_csv('experiment_results.csv', index=False)
print("实验结果已保存为 experiment_results.csv")