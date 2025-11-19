import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from old.cpt import CollaborativeStack
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load MNIST test set
transform = transforms.Compose([transforms.ToTensor()])
test_set = datasets.MNIST("../mnist_data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# load the saved model
model = torch.load("mnist_collabnet.pth", map_location=device, weights_only=False)
model.eval()

# evaluate using the same approach as in training
criterion = nn.CrossEntropyLoss()
correct = 0
total = 0
total_loss = 0.0

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.view(xb.size(0), -1).to(device)
        yb = yb.to(device)
        
        # Forward through all branches
        outs = model.forward_until(xb, upto_layer=len(model.branches) - 1)
        logits = outs[len(model.branches) - 1][0]
        
        # Calculate loss
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)  # ← KEY FIX: specify dim=1
        correct += (preds == yb).sum().item()
        total += xb.size(0)

acc = correct / total
avg_loss = total_loss / total
print(f"Test accuracy: {acc*100:.2f}%")
print(f"Test loss: {avg_loss:.4f}")