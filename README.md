# Stacked Autoencoder CollabNet

This repository contains experiments with stacked autoencoder inspired collaborative network architecture (“SAECollabNet”).  
The project was originally developed as a research prototype and is intended for experimentation and study, not as a polished library.

---

## Repository structure

- `collabNet.py`  
  Core implementation of the CollabNet architecture and training loop, including:
  - Stacked autoencoder backbone
  - Variations of optimization / mutation strategies

- `mnist_spaced.py`  
  MNIST training script using spaced training.

- `mnist_tolerance.py`  
  MNIST experiments focused on tolerance.

- `draw_mnist.py`  
  Utility for visualizing MNIST samples and reconstructions.

- `old/`  
  Legacy scripts and early experiments (kept for reference).

- `CITATION.cff`  
  Machine-readable citation file with full metadata (authors, title, DOI, abstract, etc.).

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- (Optional) GPU support via CUDA

---

## How to run

All scripts assume MNIST is available via `torchvision`.

Train a CollabNet-based model on MNIST:

```bash
python collabNet.py
```

Run one of the MNIST experiments:

```bash
python mnist_spaced.py
python mnist_tolerance.py
```

Visualize MNIST digits or reconstructions:

```bash
python draw_mnist.py
```

The exact CLI options (if any) are defined inside each script; open the file to adjust hyperparameters or paths as needed.

---

## Citation

This project includes a `CITATION.cff` file with full citation metadata.

You can cite this work using the file directly:

[➡️ View `CITATION.cff`](./CITATION.cff)
