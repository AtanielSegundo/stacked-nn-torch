# collaborative_stack.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

torch.manual_seed(0)


# Helpers
def get_activation(name: str):
    name = name.lower()
    if name in ("relu",):
        return nn.ReLU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("sigmoid",):
        return nn.Sigmoid()
    if name in ("identity", "linear", "none"):
        return nn.Identity()
    if name in ("gelu"):
        return nn.GELU()
	# default
    return nn.Tanh()


class BranchModule(nn.Module):
    """
    Single branch + collaborative combiner.
    - whi: input -> hidden
    - optional extra branch: whi_ext (input -> ext) and woh_ext (ext -> hidden contribution)
    - woh: hidden -> out (used if you want direct out-from-hidden)
    - combiner W2: maps [O_prev, H_new] -> O_new
      left block (columns 0:out_dim) can be initialized to k*I and held fixed via gradient projection.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        method: str = "M1",
        extra_dim: int = 0,
        k_identity: float = 1.0,
        activation_hidden: str = "gelu",
        activation_out: str = "identity",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.method = method
        self.extra_dim = extra_dim
        self.k_identity = float(k_identity)

        # primary branch
        self.whi = nn.Linear(in_dim, hid_dim)
        self.act_h = get_activation(activation_hidden)
        # direct hidden->out path (optional, used for weight bookkeeping)
        self.woh = nn.Linear(hid_dim, out_dim)

        # optional extra branch (M3, M4)
        if self.method in ("M3", "M4") and extra_dim > 0:
            self.whi_ext = nn.Linear(in_dim, extra_dim)
            self.act_h_ext = get_activation(activation_hidden)
            # extra -> hidden contribution
            self.woh_ext = nn.Linear(extra_dim, hid_dim)
        else:
            self.whi_ext = None
            self.act_h_ext = None
            self.woh_ext = None

        # collaborative combiner W2: in_features = out_dim + hid_dim
        self.W2 = nn.Linear(out_dim + hid_dim, out_dim)
        self.act_out = get_activation(activation_out)

        # Initialize W2 left block to k * Identity and right block zeros
        with torch.no_grad():
            self.W2.weight.zero_()
            self.W2.bias.zero_()
            n = out_dim
            if n > 0:
                eye = torch.eye(n)
                # left block: columns 0:n (corresponds to O_prev)
                self.W2.weight[:, :n] = self.k_identity * eye

        # flags for projection (which gradients to zero)
        # If True, we'll zero grads for W2.weight[:, :out_dim] after backward
        self.freeze_W2_left_block = False
        # For hiding: zero grads for first out_dim rows of whi (hidden weights)
        self.freeze_whi_first_out_rows = False

    def forward(self, x: torch.Tensor, o_prev: Optional[torch.Tensor] = None):
        """
        x: (B, in_dim)
        o_prev: (B, out_dim) or None (for first layer)
        returns: out (B, out_dim), hidden (B, hid_dim), ext_hidden (B, extra_dim or None)
        """
        # primary hidden
        hid = self.whi(x)  # (B, hid_dim)
        # extra branch contribution
        hid_ext = None
        if self.whi_ext is not None:
            ext = self.whi_ext(x)
            ext = self.act_h_ext(ext)
            hid_ext = self.woh_ext(ext)  # maps ext->hid contribution
            hid = hid + hid_ext

        hid = self.act_h(hid)

        # direct woh path (optional use, generally W2 will combine)
        # out_from_hid = self.woh(hid)

        if o_prev is None:
            # if no previous output, use zeros
            batch = x.shape[0]
            o_prev = torch.zeros(batch, self.out_dim, device=x.device, dtype=x.dtype)
        # concat [o_prev, hid]
        concat = torch.cat([o_prev, hid], dim=1)  # (B, out_dim + hid_dim)
        out = self.W2(concat)
        out = self.act_out(out)
        return out, hid, hid_ext

    def project_gradients(self):
        """
        Apply gradient projection to enforce frozen sub-blocks.
        Must be called after loss.backward() and before optimizer.step()
        """
        if self.W2.weight.grad is not None:
            if self.freeze_W2_left_block:
                # zero gradients corresponding to left block columns 0:out_dim
                self.W2.weight.grad[:, : self.out_dim].zero_()
                # optionally freeze bias part? keep bias trainable
        if self.whi.weight.grad is not None:
            if self.freeze_whi_first_out_rows:
                # rows 0:out_dim of whi correspond to first hidden rows in original code.
                # In original code these rows were kept constant; we zero their grads here.
                r = min(self.out_dim, self.whi.weight.grad.shape[0])
                if r > 0:
                    self.whi.weight.grad[:r, :].zero_()
                    if self.whi.bias is not None and self.whi.bias.grad is not None:
                        self.whi.bias.grad[:r].zero_()


class CollaborativeStack(nn.Module):
    """
    Stack of BranchModule layers.
    Provides methods:
    - add_branch
    - train_sequential_layers(...) : trains layer by layer with freeze/grad-projection
    - predict
    """

    def __init__(self, input_dim: int, out_dim: int, device: Optional[torch.device] = None):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.branches = nn.ModuleList()
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def add_branch(
        self,
        hid_dim: int,
        method: str = "M1",
        extra_dim: int = 0,
        k_identity: float = 1.0,
        activation_hidden: str = "tanh",
        activation_out: str = "identity",
    ):
        """
        Add a new branch at the end of the stack.
        """
        in_dim = self.input_dim
        b = BranchModule(
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=self.out_dim,
            method=method,
            extra_dim=extra_dim,
            k_identity=k_identity,
            activation_hidden=activation_hidden,
            activation_out=activation_out,
        ).to(self.device)
        self.branches.append(b)
        return b

    def forward_until(self, x: torch.Tensor, upto_layer: int):
        """
        Forward pass through layers 0..upto_layer inclusive (1-based index conceptually).
        Returns dictionary of per-layer (out, hid, hid_ext)
        """
        outs = {}
        o_prev = None
        input_tensor = x
        for i, layer in enumerate(self.branches[: upto_layer + 1]):
            out, hid, hid_ext = layer(input_tensor, o_prev=o_prev)
            outs[i] = (out, hid, hid_ext)
            # next o_prev is this out
            o_prev = out
            # input_tensor stays same as original input (the original code used xin for external branch)
        return outs

    def predict(self, x: np.ndarray, change: str = "Normal") -> np.ndarray:
        """
        x: numpy array (N, input_dim)
        returns: numpy array (N, out_dim)
        """
        self.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            outs = self.forward_until(xt, upto_layer=len(self.branches) - 1)
            final = outs[len(self.branches) - 1][0]
            return final.cpu().numpy()

    # Utility to set train/freeze flags for layers given a training mode for the target layer
    def _prepare_layer_flags(self, target_idx: int, mode: str):
        """
        mode in {'Normal', 'ChangeOut', 'ChangeHide'}
        - freezes previous layers (requires_grad=False)
        - configures flags on target layer for gradient projection (freeze left block etc.)
        """
        # freeze previous layers parameters
        for i, layer in enumerate(self.branches):
            for p in layer.parameters():
                p.requires_grad = False
            # reset projection flags
            layer.freeze_W2_left_block = False
            layer.freeze_whi_first_out_rows = False

        # target layer: set requires_grad True for its params by default
        target_layer = self.branches[target_idx]
        for p in target_layer.parameters():
            p.requires_grad = True

        # set projections according to mode
        if mode == "Normal":
            # no projection; train whole target layer
            pass
        elif mode == "ChangeOut":
            # keep W2 left block fixed; train W2 right and branch params
            target_layer.freeze_W2_left_block = True
            # also ensure previous layers stay frozen
        elif mode == "ChangeHide":
            # keep W2 left block fixed; and prevent updates in first out_dim rows of whi
            target_layer.freeze_W2_left_block = True
            target_layer.freeze_whi_first_out_rows = True
        else:
            raise ValueError("Unknown mode")

    def train_layer(
        self,
        layer_idx: int,
        train_loader,
        epochs: int,
        lr_hi: float = 1e-3,
        lr_oh: float = 1e-3,
        mode: str = "Normal",
        verbose: int = 0,
        project_grad_after_backward: bool = True,
    ):
        """
        Train up to layer_idx and update only parameters of branch layer_idx (with mode controlling projections).
        train_loader yields (x_np, y_np) with shapes (N, input_dim) and (N, out_dim)
        lr_hi and lr_oh can be used to create two param groups. Here we unify into one for simplicity,
        but we prepare separate groups if user passed two values in a list.
        """
        assert 0 <= layer_idx < len(self.branches)
        self.train()
        self._prepare_layer_flags(layer_idx, mode)

        # collect parameters to train (those with requires_grad True)
        params_to_train = [p for p in self.parameters() if p.requires_grad]
        opt = optim.SGD(params_to_train, lr=lr_hi)

        loss_fn = nn.MSELoss()

        # basic epoch loop
        for ep in range(1, epochs + 1):
            epoch_loss = 0.0
            batches = 0
            for xb_np, yb_np in train_loader:
                xb = torch.tensor(xb_np, dtype=torch.float32, device=self.device)
                yb = torch.tensor(yb_np, dtype=torch.float32, device=self.device)

                # forward through stack until target layer
                outs = self.forward_until(xb, upto_layer=layer_idx)
                y_pred = outs[layer_idx][0]  # (B, out_dim)
                loss = loss_fn(y_pred, yb)
                opt.zero_grad()
                loss.backward()

                # projection operations per layer
                # zero grads on frozen subblocks
                if project_grad_after_backward:
                    for i, layer in enumerate(self.branches[: layer_idx + 1]):
                        layer.project_gradients()

                # step
                opt.step()
                epoch_loss += loss.item()
                batches += 1

            if verbose and (ep % verbose == 0 or ep == 1 or ep == epochs):
                print(f"Layer {layer_idx} Ep {ep}/{epochs}  loss={epoch_loss / max(1,batches):.6f}")

    # Convenience: train sequentially all layers as in paper's fit_train routine
    def train_sequential(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        epochs_branch: int = 100,
        batch_size: int = 32,
        mode_sequence: List[str] = None,
        lr_hi: float = 1e-3,
        verbose_every: int = 10,
    ):
        """
        mode_sequence: list of modes per layer addition. If None, use 'Normal' for all.
        This function assumes branches already added to the stack.
        """
        if mode_sequence is None:
            mode_sequence = ["Normal"] * len(self.branches)
        assert len(mode_sequence) == len(self.branches)

        X, Y = train_data
        n = X.shape[0]
        indices = np.arange(n)

        # simple batch generator
        def batch_iter():
            # yields (xb, yb) numpy arrays
            for i in range(0, n, batch_size):
                idx = indices[i : i + batch_size]
                yield X[idx], Y[idx]

        # train each layer in sequence
        for layer_idx, mode in enumerate(mode_sequence):
            print(f"--- Training layer {layer_idx} mode={mode} ---")
            # for epochs_branch epochs
            for ep in range(1, epochs_branch + 1):
                epoch_loss = 0.0
                batches = 0
                for xb, yb in batch_iter():
                    xb_t = torch.tensor(xb, dtype=torch.float32, device=self.device)
                    yb_t = torch.tensor(yb, dtype=torch.float32, device=self.device)
                    # prepare flags and optimizer per batch call: delegate to train_layer logic but here we do whole epoch manually
                    # simpler: call train_layer with batch-wise dataloader wrapper
                    pass
                # Do one epoch using train_layer but with full epoch as collection
                # Build a tiny loader that yields full dataset in batches
                self.train_layer(
                    layer_idx,
                    train_loader=batch_iter(),
                    epochs=1,
                    lr_hi=lr_hi,
                    lr_oh=lr_hi,
                    mode=mode,
                    verbose=0,
                    project_grad_after_backward=True,
                )
                if ep % verbose_every == 0 or ep == 1 or ep == epochs_branch:
                    # compute train loss for reporting
                    preds = self.predict(X)
                    mse = float(np.mean((preds - Y) ** 2))
                    print(f"Layer {layer_idx} epoch {ep}/{epochs_branch} mse={mse:.6f}")


# -----------------------------
# Demo / unit tests
# -----------------------------
def demo_identity_preservation():
    """
    Tests: add first branch, produce baseline outputs.
           add second branch with W2.right zeros and k=1 => new predict equals baseline.
    """
    print("Demo: identity preservation test")
    # toy dataset
    N = 200
    in_dim = 4
    out_dim = 2
    X = np.random.randn(N, in_dim).astype(np.float32)
    # target is linear function of inputs
    trueW = np.random.randn(in_dim, out_dim).astype(np.float32)
    Y = X.dot(trueW)

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    model = CollaborativeStack(input_dim=in_dim, out_dim=out_dim, device=device)
    # add first branch
    model.add_branch(hid_dim=8, method="M3", extra_dim=0, k_identity=1.0)
    # quick random init training to make branch non-trivial
    # we simulate training by small SGD steps directly on first branch so we have a baseline
    def quick_train_first():
        opt = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        loss_fn = nn.MSELoss()
        for _ in range(50):
            xb = torch.tensor(X, dtype=torch.float32, device=device)   # <<< add device=device
            yb = torch.tensor(Y, dtype=torch.float32, device=device)   # <<< add device=device
            outs = model.forward_until(xb, upto_layer=0)
            pred = outs[0][0]
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    quick_train_first()
    baseline = model.predict(X)

    # add second branch with identity left-block and zeros right-block
    b2 = model.add_branch(hid_dim=6, method="M3", extra_dim=0, k_identity=1.0)
    # ensure W2 left block is identity and right block zeros (already done in constructor)
    # But if k !=1 then set to 1 for identity preservation test
    with torch.no_grad():
        b2.W2.weight[:, :out_dim] = torch.eye(out_dim)

    # test before training: prediction of full stack should equal baseline because W2 right block zeros and left block identity
    pred_new = model.predict(X)
    max_diff = float(np.max(np.abs(pred_new - baseline)))
    print(f"Max abs difference before training new branch: {max_diff:.10f}")
    assert max_diff < 1e-5, "Identity preservation test failed."

    print("Identity preservation test passed.")

def demo_train_with_new_branch():
    print("Demo: train with new branch improves MSE on simple regression")
    N = 10900
    in_dim = 20
    out_dim = 5
    X = np.random.randn(N, in_dim).astype(np.float32)
    W_true = np.array([[2.0], [-1.0], [0.5]], dtype=np.float32)
    Y = (X.dot(W_true) + 0.1 * np.random.randn(N, 1)).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CollaborativeStack(input_dim=in_dim, out_dim=out_dim, device=device)
    model.add_branch(hid_dim=6, method="M3", extra_dim=0, k_identity=1.0)

    # train first branch alone (quick)
    loader = [(X, Y)]
    losses_first = []
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss_fn = nn.MSELoss()

    for ep in range(100):
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        yb = torch.tensor(Y, dtype=torch.float32, device=device)
        outs = model.forward_until(xb, upto_layer=0)
        pred = outs[0][0]
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses_first.append(loss.item())

    mse_before = float(np.mean((model.predict(X) - Y) ** 2))
    print(f"MSE after first branch train: {mse_before:.6f}")

    # add second branch
    model.add_branch(hid_dim=8, method="M3", extra_dim=0, k_identity=1.0)
    with torch.no_grad():
        model.branches[-1].W2.weight[:, :out_dim] = torch.eye(out_dim)

    mse_preserve = float(np.mean((model.predict(X) - Y) ** 2))
    print(f"MSE right after adding branch (should equal previous): {mse_preserve:.6f}")

    # train second branch
    loader = [(X, Y)]
    losses_second = []
    loss_fn = nn.MSELoss()
    model._prepare_layer_flags(1, "ChangeOut")
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    opt = optim.SGD(params_to_train, lr=1e-2)

    for ep in range(300):
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        yb = torch.tensor(Y, dtype=torch.float32, device=device)
        outs = model.forward_until(xb, upto_layer=1)
        pred = outs[1][0]
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        model.branches[1].project_gradients()
        opt.step()
        losses_second.append(loss.item())

    mse_after = float(np.mean((model.predict(X) - Y) ** 2))
    print(f"MSE after training second branch: {mse_after:.6f}")

    # --- plot ---
    plt.figure(figsize=(10,5))
    plt.plot(losses_first, label="Branch 1 training loss")
    plt.plot(range(len(losses_first), len(losses_first)+len(losses_second)), losses_second, label="Branch 2 training loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Collaborative Stack Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    demo_identity_preservation()
    demo_train_with_new_branch()