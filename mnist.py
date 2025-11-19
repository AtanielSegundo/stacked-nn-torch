# mnist_collabnet.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from collabNet import SAECollabNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Hiperparametros
input_dim = 28 * 28
num_classes = 10
batch_size = 256
initial_hid = 64
new_branch_hid = 32
extra_hid = 16
max_epochs = 50
patience = 5
min_delta = 1e-3
max_branches = 32
lr_initial = 5e-3
lr_branch = lr_initial

data_dir = "./mnist_data"
os.makedirs(data_dir, exist_ok=True)
transform = transforms.Compose([transforms.ToTensor()])
full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

val_size = 5000
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

model = SAECollabNet(input_dim=input_dim, 
                  first_hidden=initial_hid, 
                  first_out=num_classes,
                  hidden_activation=nn.GELU(), 
                  out_activation=nn.Identity(),
                  device=device
                  )

current_top = 0               
branches_added = 1
history = {"train_loss": [], "val_loss": [], "val_acc": [], "branch_changes": []}

def evaluate(net, loader, criterion):
    net.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.view(xb.size(0), -1).to(device)
            yb = yb.to(device)
            logits, _, _ = net.forward(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=lr_initial)
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
epochs_since_improve = 0

for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for xb, yb in train_loader:
        xb = xb.view(xb.size(0), -1).to(device)
        yb = yb.to(device)

        logits, _, _ = model.forward(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
        
    model.step_all_etas()
    train_loss = running_loss / n_samples
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch} | Layer {current_top} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    if val_loss + min_delta < best_val_loss:
        best_val_loss = val_loss
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1

    if epochs_since_improve >= patience and branches_added < max_branches:
        print(f"Validation loss saturated for {epochs_since_improve} epochs. Adding new layer.")
        model.add_layer(
            hidden_dim=new_branch_hid,
            out_dim=num_classes,
            extra_dim=extra_hid,
            k=1.0,
            mutation_mode="Hidden",
            target_fn=nn.GELU(),
            eta=0.0,
            eta_increment= 1/max_epochs,
            hidden_activation=nn.GELU(),
		    extra_activation=nn.Identity(),
            out_activation=nn.Identity(),
        )

        current_top = len(model.layers) - 1
        branches_added += 1

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=lr_branch)

        history["branch_changes"].append({"epoch": epoch, "new_layer": current_top})
        best_val_loss = float("inf")
        epochs_since_improve = 0

    if epoch == max_epochs:
        print("Reached max epochs.")

save_path = "mnist_collabnet_state.pth"
torch.save(model,save_path)
print(f"Model state_dict saved to {save_path}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(history["train_loss"], label="Train Loss", color="tab:blue")
ax1.plot(history["val_loss"], label="Val Loss", color="tab:orange")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

val_acc_percent = [v * 100 for v in history["val_acc"]]
ax2.plot(val_acc_percent, label="Val Accuracy (%)", color="tab:green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True)

# MARCANDO AS EPOCAS EM QUE CAMADAS FORAM ADICIONADAS
if history["branch_changes"]:
    vmax = max(history["val_loss"]) if len(history["val_loss"]) > 0 else 1.0
    for bc in history["branch_changes"]:
        e = bc["epoch"] - 1
        ax1.axvline(e, color="red", linestyle="--", alpha=0.7)
        ax2.axvline(e, color="red", linestyle="--", alpha=0.7)
        ax1.text(e + 0.3, vmax * 0.95, f"layer {bc['new_layer']}", rotation=90, color="red", fontsize=8)

plt.tight_layout()
plt.show()