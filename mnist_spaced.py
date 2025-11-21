# mnist_spaced.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from collabNet import SAECollabNet, MutationMode

def evaluate(net, loader, criterion, device):
    net.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.view(xb.size(0), -1).to(device)
            yb = yb.to(device)

            out = net(xb)
            # handle both nn.Sequential (tensor) and SAECollabNet (tuple)
            if isinstance(out, tuple) or isinstance(out, list):
                logits = out[0]
            else:
                logits = out

            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total

if __name__ == "__main__":
    # HYPERPARAMETERS
    seed = 0
    max_epochs = 100
    num_of_branchs = 8
    insert_epoch_mod = max_epochs // num_of_branchs

    lr_initial = 1e-4
    lr_branch  = 5e-4

    input_dim = 28 * 28
    num_classes = 10
    batch_size = 512

    initial_hid = 256
    new_branch_hid = 128
    extra_hid = 64
    standard_hid = new_branch_hid + extra_hid

    # ACTIVATION FUNCTIONS
    hidden_activation = nn.GELU()
    out_activation    = nn.Identity()
    extra_activation  = nn.Sigmoid()
    mutation_mode     = MutationMode.Hidden
    target_activation = nn.GELU()

    # DEVICE SETTING
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # SEED SETTING
    torch.manual_seed(seed)
    np.random.seed(seed)

    # DATASET LOADING
    data_dir = "./mnist_data"
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])

    val_size   = 5000
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set   = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_size = len(full_train) - val_size
    train_set, val_set   = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # STANDARD SEQUENTIAL MODEL (baseline with roughly comparable capacity)
    modules = []
    modules.append(nn.Linear(input_dim, initial_hid))
    modules.append(hidden_activation)
    modules.append(nn.Linear(initial_hid, standard_hid))
    modules.append(hidden_activation)
    # add internal standard_hid layers to match number of branch insertions
    for i in range(num_of_branchs):
        modules.append(nn.Linear(standard_hid, standard_hid))
        modules.append(target_activation)  # nonlinearity between stacked layers
    modules.append(nn.Linear(standard_hid, num_classes))

    standard_model = nn.Sequential(*modules).to(device)
    standard_params    = [p for p in standard_model.parameters() if p.requires_grad]
    standard_optimizer = optim.Adam(standard_params, lr=lr_initial)
    standard_criterion = nn.CrossEntropyLoss()

    # COLABORATIVE NETWORK
    colab_model = SAECollabNet(
                    input_dim=input_dim,
                    first_hidden=initial_hid,
                    first_out=num_classes,
                    hidden_activation=hidden_activation,
                    out_activation=out_activation,
                    device=device,
                    )
    colab_model.to(device)

    colab_params    = [p for p in colab_model.parameters() if p.requires_grad]
    colab_optimizer = optim.Adam(colab_params, lr=lr_initial)
    colab_criterion = nn.CrossEntropyLoss()

    # TRACK TRAINING VARIABLES
    current_top = 0
    branches_added = 1
    colab_history = {"train_loss": [], "val_loss": [], "val_acc": [], "branch_changes": []}
    standard_history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # TRAINING PHASE
    colab_model.train()
    standard_model.train()
    for epoch in range(1, max_epochs + 1):

        # INSERTING BRANCH (spaced insertion)
        if epoch % insert_epoch_mod == 0 and (branches_added < (num_of_branchs + 1)):
            colab_model.add_layer(
                hidden_dim=new_branch_hid,
                out_dim=num_classes,
                extra_dim=extra_hid,
                k=1.0,
                mutation_mode=mutation_mode,
                target_fn=target_activation,
                eta=0.0,
                eta_increment=1 / insert_epoch_mod,
                hidden_activation=hidden_activation,
                extra_activation=extra_activation,
                out_activation=out_activation,
            )

            current_top = len(colab_model.layers) - 1
            branches_added += 1

            colab_params = [p for p in colab_model.parameters() if p.requires_grad]
            colab_optimizer = optim.Adam(colab_params, lr=lr_branch)
            colab_history["branch_changes"].append({"epoch": epoch, "new_layer": current_top})

        standard_running_loss = 0.0
        colab_running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.view(xb.size(0), -1).to(device)
            n_samples += xb.size(0)
            yb = yb.to(device)

            # STANDARD MODEL
            logits_std = standard_model(xb)  # nn.Sequential returns tensor
            loss_std = standard_criterion(logits_std, yb)
            standard_optimizer.zero_grad()
            loss_std.backward()
            standard_optimizer.step()

            standard_running_loss += loss_std.item() * xb.size(0)

            # COLAB MODEL
            logits_colab, _, _ = colab_model(xb)
            loss_colab = colab_criterion(logits_colab, yb)
            colab_optimizer.zero_grad()
            loss_colab.backward()
            colab_optimizer.step()

            colab_running_loss += loss_colab.item() * xb.size(0)

        # advance betas/etas in collab model
        colab_model.step_all_etas()

        # record collab metrics
        train_loss_colab = colab_running_loss / n_samples
        val_loss_colab, val_acc_colab = evaluate(colab_model, val_loader, colab_criterion, device)
        colab_history["train_loss"].append(train_loss_colab)
        colab_history["val_loss"].append(val_loss_colab)
        colab_history["val_acc"].append(val_acc_colab)

        # record standard metrics
        train_loss_std = standard_running_loss / n_samples
        val_loss_std, val_acc_std = evaluate(standard_model, val_loader, standard_criterion, device)
        standard_history["train_loss"].append(train_loss_std)
        standard_history["val_loss"].append(val_loss_std)
        standard_history["val_acc"].append(val_acc_std)

        # logging
        print(f"Epoch {epoch} | Layer {current_top} | Collab train {train_loss_colab:.4f} val {val_loss_colab:.4f} acc {val_acc_colab:.4f} | Std train {train_loss_std:.4f} val {val_loss_std:.4f} acc {val_acc_std:.4f}")
        colab_model.debug_eta_status()

    # final evaluation on test set
    test_loss_colab, test_acc_colab = evaluate(colab_model, test_loader, colab_criterion, device)
    test_loss_std, test_acc_std     = evaluate(standard_model, test_loader, standard_criterion, device)
    print(f"\nFinal test — Collab: loss {test_loss_colab:.4f} acc {test_acc_colab:.4f} | Standard: loss {test_loss_std:.4f} acc {test_acc_std:.4f}")

    # save models
    os.makedirs("saved_models", exist_ok=True)
    torch.save(colab_model.state_dict(), "saved_models/colab_model_state.pth")
    torch.save(standard_model.state_dict(), "saved_models/standard_model_state.pth")
    print("Models saved to saved_models/")

    # PLOTS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(colab_history["train_loss"], label="Collab Train Loss")
    ax1.plot(standard_history["train_loss"], label="Standard Train Loss", linestyle="dashed")
    ax1.plot(colab_history["val_loss"], label="Collab Val Loss")
    ax1.plot(standard_history["val_loss"], label="Standard Val Loss", linestyle="dashed")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    val_acc_percent_colab = [v * 100 for v in colab_history["val_acc"]]
    val_acc_percent_std   = [v * 100 for v in standard_history["val_acc"]]

    ax2.plot(val_acc_percent_colab, label="Collab Val Accuracy (%)")
    ax2.plot(val_acc_percent_std, label="Standard Val Accuracy (%)", linestyle="dashed")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # MARK LAYER INSERTIONS
    if colab_history["branch_changes"]:
        vmax = max(colab_history["val_loss"]) if len(colab_history["val_loss"]) > 0 else 1.0
        for bc in colab_history["branch_changes"]:
            e = bc["epoch"] - 1
            ax1.axvline(e, color="red", linestyle="--", alpha=0.7)
            ax2.axvline(e, color="red", linestyle="--", alpha=0.7)
            ax1.text(e + 0.3, vmax * 0.95, f"layer {bc['new_layer']}", rotation=90, color="red", fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.savefig("saved_models/mnist_")