# Neural-network-based Dynamic-optical-coherence-tomography  library by computational optics group (COG NN-DOCT library) 
The COG NN-DOCT library is a Python library which provides the open-source codes for generating dynamic optical coherence tomography (DOCT) contrasts from a a small number of OCT frames.

## NNASDOCT: Dual-input Burst NN

TensorFlow/Keras 2.6 implementation of the proposed dual-input burst neural network
for amplitude-spectrum dynamic optical coherence tomography (AS-DOCT). The model
uses a 16-frame multi-burst OCT sequence and its pseudo-amplitude spectrum to
generate low-, middle-, and high-frequency AS-DOCT images.

The detailed methods of NN-AS-DOCT can be found in [Liu et al., arXiv:2608.09171v1](https://arxiv.org/abs/2608.09171)).

Files
-----

- `NNmodel.py`: the network architecture, constructed using `LSTM_3dUnet()`.
- `NNtrain.py`: data preprocessing and training, with user settings at the top.
- `early_stop_tool.py`: the early-stopping utility used during training.
- `requirements.txt`: Python dependencies.
- `ExampleDataset/`: the small example dataset described below.

Example dataset
---------------

The example dataset contains **40 training patches and 10 validation patches**
extracted from the original training and validation datasets, respectively.
OCT inputs and their AS-DOCT targets are paired in the same sample order.

| File | Shape | Values |
| --- | --- | --- |
| `train_oct.npy` | `(40, 16, 64, 64)` | Linear OCT intensity |
| `train_as.npy` | `(40, 3, 64, 64)` | Linear AS-DOCT amplitude |
| `val_oct.npy` | `(10, 16, 64, 64)` | Linear OCT intensity |
| `val_as.npy` | `(10, 3, 64, 64)` | Linear AS-DOCT amplitude |

For OCT arrays, the axes are patch, frame, depth, and fast-scan position.
For AS-DOCT arrays, the axes are patch, frequency band, depth, and fast-scan
position. The three frequency bands are ordered **low, middle, high**.

This subset is provided to demonstrate the training workflow. It is not the full
dataset used to obtain the results reported in the paper.

Run
---


Use a Python 3.7–3.9 environment with TensorFlow/Keras 2.6. Run the following
commands from the `NNASDOCT` directory.

1. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. To use the provided dataset, set the input directory in `NNtrain.py` to:

   ```python
   DATA_DIR = BASE_DIR / "ExampleDataset"
   ```

   The four default input filenames already match the example dataset. The output
   directory (`root_`) and training parameters can also be changed at the top of
   the script. For a brief software demonstration, `MAX_EPOCHS` can be set to `1`.

3. Start training:

   ```bash
   python NNtrain.py
   ```

The script applies a square root to OCT intensity, computes the full 16-point FFT
magnitude for the second input, and applies `log10` to AS-DOCT targets. Input files
should contain linear values, without these transformations applied beforehand.

Outputs
-------

By default, training results are saved in `output/` beside `NNtrain.py`:

- `model.h5`: the model with the lowest validation loss.
- `epoch_*/model.h5`: the model saved at each epoch.
- `loss/`: total and per-band loss histories, training time, and the learning curve.

Network predictions are **log10 AS-DOCT amplitudes** in low/middle/high order.
The three model outputs correspond to frequency bands, rather than RGB display
channels.

Acknowledgment
--------------

The early-stopping utility is adapted from
[Bjarten/early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch).

License
-------
This library is licensed under either of the following options.
1. [GNU lesser general public license version 3 (LGPLv3)](LICENSE_GnuLGPLv3.md).
2. Any licenses except for GNU LGPLv3 as far as the authors and the uses agree for it. It may include business licenses, closed-source licenses, and others. 
 
Option 1 (GNU LGPLv3) can be selected without notifying the authors. (However, it is recommended to cite this web-site and the proper reference papers listed in "available DOCT contrasts" sections when you publish research papers using this library.)
If you want to select any other licensing conditions except for GNU LGPLv3 (i.e., Option 2), please contact the corresponding author (Yoshiaki Yasuno, University of Tsukuba, <yoshiaki.yasuno@cog-labs.org>) to obtain an explicit agreement.
