# NNASDOCT: proposed Dual-input Burst NN of TensorFlow/Keras 2.6 implementation.

## Files

- `NNmodel.py`: the Dual-input Burst architectur.
- `NNtrain.py`: preprocessing and training, with user settings at the top.
- `early_stop_tool.py`: the early-stopping class used by training.


## Run

1. Use a Python 3.7–3.9 environment with TensorFlow/Keras 2.6.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Set `DATA_DIR`, the four data filenames, and `root_` at the top of the training script.
4. Run: `python NNtrain.py`.

Defaults are relative to the script: inputs in `data/`, results in `output/`.
The small experimental demonstration dataset has not been added yet.

OCT arrays contain linear **intensity**, shape `(N, 16, H, W)`.
AS targets contain linear **amplitude**, shape `(N, 3, H, W)`, ordered low/middle/high.
The original training uses 64 x 64 patches. The script itself applies square root
to OCT intensity and log10 to AS targets; do not pre-apply those transforms.

The architecture, nested FFT loops, loss, gradient calculation, per-epoch Adam
reinitialization, early stopping, and checkpoint logic are preserved.
Network outputs are log10 AS amplitudes, ordered low/middle/high.


Early stopping is adapted from Bjarten/early-stopping-pytorch; see the source
attribution in `early_stop_tool.py` 
