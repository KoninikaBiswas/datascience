import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

digits = load_digits()
X = digits.images.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=SEED
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train)[:, None], torch.tensor(y_train)),
    batch_size=64, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val)[:, None], torch.tensor(y_val)),
    batch_size=128
)

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*2*2, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = DigitCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-4
)

best_loss = float("inf")
best_state = None
wait = 0

for epoch in range(40):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            total_loss += criterion(model(xb), yb).item() * len(yb)
            n += len(yb)
    val_loss = total_loss / n

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    if wait >= 6:
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    test_x = torch.tensor(X_test)[:, None]
    predictions = model(test_x).argmax(1).numpy()

print("Test accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, digits=3))
