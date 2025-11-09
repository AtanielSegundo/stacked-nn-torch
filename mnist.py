# mnist.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from cpt import CollaborativeStack  # your cpt.py must be in same folder

# Config
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Hyperparameters
input_dim = 28 * 28
num_classes = 10
batch_size = 128
initial_hid = 64
new_branch_hid = 64
max_epochs = 50              # overall training cap
patience = 3
min_delta = 5e-3
max_branches = 32
lr_initial = 1e-3
lr_branch = lr_initial

# Data
data_dir = "./mnist_data"
os.makedirs(data_dir, exist_ok=True)
transform = transforms.Compose([transforms.ToTensor()])
full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

# split train -> train/val
val_size = 5000
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

# Model
model = CollaborativeStack(input_dim=input_dim, out_dim=num_classes, device=device)
# add first branch
model.add_branch(hid_dim=initial_hid, method="M1", extra_dim=0, k_identity=1.0)

# Utility: evaluate on dataset (returns avg loss and accuracy)
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.view(xb.size(0), -1).to(device)
            yb = yb.to(device)
            outs = model.forward_until(xb, upto_layer=len(model.branches) - 1)
            logits = outs[len(model.branches) - 1][0]
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total

# Start training state
current_top = 0
branches_added = 1
history = {"train_loss": [], "val_loss": [], "val_acc": [], "branch_changes": []}

# Prepare optimizer for initial branch
# Freeze previous (none) and unfreeze target
model._prepare_layer_flags(current_top, "Normal")
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=lr_initial)
criterion = nn.CrossEntropyLoss()

# Early-saturation tracking
best_val_loss = float("inf")
epochs_since_improve = 0

# Training loop
for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for xb, yb in train_loader:
        xb = xb.view(xb.size(0), -1).to(device)
        yb = yb.to(device)

        # forward up to current top
        outs = model.forward_until(xb, upto_layer=current_top)
        logits = outs[current_top][0]
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()

        # project grads for frozen parts (if any)
        # apply projection for layers up to current_top (only top may have flags)
        for i, layer in enumerate(model.branches[: current_top + 1]):
            layer.project_gradients()

        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)

    train_loss = running_loss / n_samples
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch} | Branch {current_top} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    # check improvement
    if val_loss + min_delta < best_val_loss:
        best_val_loss = val_loss
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1

    # Branch-add condition
    if epochs_since_improve >= patience and branches_added < max_branches:
        print(f"Validation loss saturated for {epochs_since_improve} epochs. Adding new branch.")
        # add branch
        new_layer = model.add_branch(
			hid_dim=new_branch_hid,
			method="M4",      # adds extra branch + learnable k
			extra_dim=32,     # strong auxiliary path
			k_identity=1.0
		)
        # force identity left-block to preserve behavior immediately
        with torch.no_grad():
            new_layer.W2.weight[:, :num_classes] = torch.eye(num_classes, device=new_layer.W2.weight.device)

        # switch to train new branch only, freezing previous
        current_top = len(model.branches) - 1
        branches_added += 1
        # prepare flags and optimizer
        model._prepare_layer_flags(current_top, "ChangeOut")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=lr_branch)

        # log branch change moment
        history["branch_changes"].append({"epoch": epoch, "new_branch": current_top})
        # reset patience tracking
        best_val_loss = float("inf")
        epochs_since_improve = 0

    # optional stopping if val loss low and no more branches desired
    if epoch == max_epochs:
        print("Reached max epochs.")

# Final evaluation on test
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"TEST: loss {test_loss:.4f}, acc {test_acc:.4f}")

save_path = "mnist_collabnet.pth"
torch.save(model, save_path)
print(f"Model saved to {save_path}")

# Plot loss and branch change markers
plt.figure(figsize=(10,5))
plt.plot(history["train_loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
for bc in history["branch_changes"]:
    e = bc["epoch"] - 1  # 0-index for plotting
    plt.axvline(e, color="red", linestyle="--", alpha=0.7)
    plt.text(e+0.2, max(history["val_loss"])*0.95, f"branch {bc['new_branch']}", rotation=90, color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
