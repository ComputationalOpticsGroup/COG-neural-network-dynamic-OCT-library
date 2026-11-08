"""Train the Dual-input Burst NN; edit the user settings below before running."""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import time
from NNmodel import LSTM_3dUnet
from early_stop_tool import EarlyStopping
# User settings: edit paths and training parameters here.
# Paths are relative to this script, so moving the repository needs no drive-letter edits.
# Change the four .npy paths below to use your prepared training/validation patches.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
root_ = BASE_DIR / "output"
TRAIN_OCT_PATH = DATA_DIR / "train_oct.npy"
TRAIN_AS_PATH = DATA_DIR / "train_as.npy"
VAL_OCT_PATH = DATA_DIR / "val_oct.npy"
VAL_AS_PATH = DATA_DIR / "val_as.npy"

# Training defaults used by the original implementation; early stopping may end sooner.
# Epoch indices start at 0: the first learning rate is 0.0005 / 1 + 0.0001 = 0.0006.
MAX_EPOCHS = 500
BATCH_SIZE = 1
LEARNING_RATE_SCALE = 0.0005
LEARNING_RATE_OFFSET = 0.0001
p = 5  # Stop after 5 consecutive epochs without validation-loss improvement.

if not os.path.exists(root_):
    os.makedirs(root_)

# Decay only the scale term; keep the original additive learning-rate offset.
def lr_decay(ini, epoch):
    return (ini / (epoch + 1)) + LEARNING_RATE_OFFSET

# OCT input: linear intensity, shape (N, 16, H, W).
# AS target: linear amplitude, shape (N, 3, H, W), ordered low/middle/high.
# Keep the original float32 cast, square root, and log10 transformations.
# Load already selected 16-frame multi-burst inputs, not an entire 512-frame sequence.
# Ground truth is supplied separately; this script does not compute AS-DOCT from 512 frames.
# AS amplitudes must be positive before log10. Do not supply already logged RGB/display data.
OCT_train = np.load(TRAIN_OCT_PATH).astype(np.float32)
AS_train = np.log10(np.load(TRAIN_AS_PATH).astype(np.float32))
OCT_val = np.load(VAL_OCT_PATH).astype(np.float32)
AS_val= np.log10(np.load(VAL_AS_PATH).astype(np.float32))
# Convert linear OCT intensity to amplitude before forming either model input.
OCT_train = np.sqrt(OCT_train)
# Compute the full 16-point FFT magnitude; no padding, shifting, or normalization.
Train_spectrum = np.zeros_like(OCT_train)
# i indexes patches, j depth pixels, and k fast-scan pixels. Each FFT uses one pixel's
# 16-value amplitude sequence. Keep the full magnitude vector in its original FFT order.
for i in range(OCT_train.shape[0]):
    for j in range(OCT_train.shape[2]):
        for k in range(OCT_train.shape[3]):
            OCT_sqrt_temp = OCT_train[i, :, j, k]
            Train_spectrum[i, :, j, k] = np.absolute(np.fft.fft(OCT_sqrt_temp, axis=0))

# Apply identical preprocessing to validation data; no separate normalization is used.
OCT_val = np.sqrt(OCT_val)
Val_spectrum = np.zeros_like(OCT_val)
for i in range(OCT_val.shape[0]):
    for j in range(OCT_val.shape[2]):
        for k in range(OCT_val.shape[3]):
            OCT_sqrt_temp = OCT_val[i, :, j, k]
            Val_spectrum[i, :, j, k] = np.absolute(np.fft.fft(OCT_sqrt_temp, axis=0))

# Add the singleton feature channel: (N, 16, H, W) -> (N, 16, H, W, 1).
# Targets remain (N, 3, H, W); individual target channels are expanded in each step.
OCT_train = np.expand_dims(OCT_train, axis=-1)
OCT_val = np.expand_dims(OCT_val, axis=-1)

Train_spectrum = np.expand_dims(Train_spectrum, axis=-1)
Val_spectrum= np.expand_dims(Val_spectrum, axis=-1)

# Keep OCT input, FFT input, and the matching target together as one dataset element.
# Shuffling later therefore preserves the correspondence between all three arrays.
train_dataset = tf.data.Dataset.from_tensor_slices((OCT_train, Train_spectrum, AS_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((OCT_val,Val_spectrum, AS_val))

# Accumulate mean total/per-band losses over each epoch, then reset the accumulators.
# The legacy 'test_loss' names below refer to validation, not a separate test dataset.
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

train_loss_low = tf.keras.metrics.Mean('train_loss_low', dtype=tf.float32)
test_loss_low = tf.keras.metrics.Mean('test_loss_low', dtype=tf.float32)

train_loss_middle = tf.keras.metrics.Mean('train_loss_middle', dtype=tf.float32)
test_loss_middle = tf.keras.metrics.Mean('test_loss_middle', dtype=tf.float32)

train_loss_high = tf.keras.metrics.Mean('train_loss_high', dtype=tf.float32)
test_loss_high = tf.keras.metrics.Mean('test_loss_high', dtype=tf.float32)

# One training batch: forward prediction, three-band MSE, gradients, and Adam update.
# x_train1/x_train2: (B, 16, H, W, 1); y_train: (B, 3, H, W), in log10 amplitude.
def train_step(model, optimizer, x_train1, x_train2, y_train, epoch_, item):
  # Track operations needed to differentiate the loss with respect to model weights.
  with tf.GradientTape() as tape:
    prelow, premiddle, prehigh = model((x_train1, x_train2), training=True)

    y_low = tf.expand_dims(y_train[:, 0, :, :], axis = -1)
    y_middle = tf.expand_dims(y_train[:, 1, :, :], axis = -1)
    y_high = tf.expand_dims(y_train[:, 2, :, :], axis = -1)

    # Split targets in low/middle/high order and match each prediction's final channel axis.
    # Keras MSE reduces that singleton channel axis; each loss still has shape (B, H, W).
    losslow = keras.losses.mse(y_low, prelow)
    lossmiddle = keras.losses.mse(y_middle, premiddle)
    losshigh = keras.losses.mse(y_high, prehigh)
    # Average across bands; preserve the original per-pixel loss map for gradients.
    loss = (losslow + lossmiddle + losshigh)/3

  # Preserve the original gradient of the unreduced loss map: TensorFlow sums its elements.
  # Replacing loss with reduce_mean(loss) here would change the gradient scale.
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  # Report scalar mean losses for logging; this does not change the gradient above.
  train_loss.update_state(loss)
  train_loss_low.update_state(losslow)
  train_loss_middle.update_state(lossmiddle)
  train_loss_high.update_state(losshigh)
  # Show running mean batch losses only during the first epoch.
  if epoch_ == 0:
      template = 'item_ {}, train_Loss: {}, train_Loss_low: {}, train_Loss_middle: {}, train_Loss_high: {}'
      print(template.format(item,
                            train_loss.result(),
                            train_loss_low.result(),
                            train_loss_middle.result(),
                            train_loss_high.result()))

# One validation batch, with training=False and no gradient/optimizer update.
# These losses monitor generalization and select the best checkpoint.
def test_step(model, x_val1, xval2, y_val):
    prelow, premiddle, prehigh = model((x_val1, xval2), training=False)

    y_low = tf.expand_dims(y_val[:, 0, :, :], axis = -1)
    y_middle = tf.expand_dims(y_val[:, 1, :, :], axis = -1)
    y_high = tf.expand_dims(y_val[:, 2, :, :], axis = -1)

    # Split targets in low/middle/high order and match each prediction's final channel axis.
    # Keras MSE reduces that singleton channel axis; each loss still has shape (B, H, W).
    losslow = keras.losses.mse(y_low, prelow)
    lossmiddle = keras.losses.mse(y_middle, premiddle)
    losshigh = keras.losses.mse(y_high, prehigh)
    loss = (losslow + lossmiddle + losshigh)/3

    test_loss.update_state(loss)
    test_loss_low.update_state(losslow)
    test_loss_middle.update_state(lossmiddle)
    test_loss_high.update_state(losshigh)
# Construct the same two-input, three-output architecture defined in NNmodel.py.
model = LSTM_3dUnet()
model.summary()

# Monitor validation loss; EarlyStopping saves each new best model as output/model.h5.
# Equal loss is not an improvement; stop after p consecutive non-improving epochs.
early_stop = EarlyStopping(patience=p, root=root_)
# Store per-epoch scalar losses before resetting metrics, for text files and the curve.
train_losses = []
validation_losses = []

train_losses_LowFre = []
validation_losses_LowFre = []

train_losses_MiddleFre = []
validation_losses_MiddleFre = []

train_losses_HighFre = []
validation_losses_HighFre = []

# Save the best validation checkpoint and each epoch's model, as in the original.
# This timer includes training, validation, and checkpoint saving inside the loop.
# It excludes loading/preprocessing above and plotting below; it is not inference time.
time1 = time.time()

for epoch in range(MAX_EPOCHS):

    # Recreate Adam each epoch, exactly as in the original training script.
    optimizer = tf.keras.optimizers.Adam(lr_decay(LEARNING_RATE_SCALE, epoch))
    iiii = 0

    # Shuffle all training patches each epoch. Each OCT/FFT/target tuple stays paired.
    for (x_train1, x_train2, y_train) in train_dataset.shuffle(OCT_train.shape[0]).batch(BATCH_SIZE):
        iiii = iiii + 1
        train_step(model, optimizer, x_train1, x_train2, y_train, epoch, iiii)
    # Evaluate validation data in fixed order, after all training updates in this epoch.
    for (x_val1, x_val2, y_val) in validation_dataset.batch(BATCH_SIZE):
        test_step(model, x_val1, x_val2, y_val)
    # Capture epoch metrics before reset_states() clears their accumulated values.
    train_losses.append(train_loss.result())
    train_losses_LowFre.append(train_loss_low.result())
    train_losses_MiddleFre.append(train_loss_middle.result())
    train_losses_HighFre.append(train_loss_high.result())

    validation_losses.append(test_loss.result())
    validation_losses_LowFre.append(test_loss_low.result())
    validation_losses_MiddleFre.append(test_loss_middle.result())
    validation_losses_HighFre.append(test_loss_high.result())
    template = 'Epoch {}, train_Loss: {}, train_Loss_low: {}, train_Loss_middle: {}, train_Loss_high: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_loss_low.result(),
                          train_loss_middle.result(),
                          train_loss_high.result()))
    template = 'test_Loss: {}, test_Loss_low: {}, test_Loss_middle: {}, test_Loss_high: {}'
    print(template.format(test_loss.result(),
                          test_loss_low.result(),
                          test_loss_middle.result(),
                          test_loss_high.result()))
    # Use validation loss, not training loss, to decide whether to save a new best model.
    early_stop(test_loss.result(), model)
    # Clear all eight metrics so the next epoch starts with fresh running means.
    train_loss.reset_states()
    train_loss_low.reset_states()
    train_loss_middle.reset_states()
    train_loss_high.reset_states()
    test_loss.reset_states()
    test_loss_low.reset_states()
    test_loss_middle.reset_states()
    test_loss_high.reset_states()
    # Keep a complete model for every epoch, separately from the best model.h5 checkpoint.
    tempmodelfolder = os.path.join(root_, 'epoch_{}'.format(epoch + 1))
    if not os.path.exists(tempmodelfolder):
        os.makedirs(tempmodelfolder)
    model.save(os.path.join(tempmodelfolder, 'model.h5'))

    # As in the original loop, save the current epoch before honoring the stop flag.
    if early_stop.early_stop:
        print('early_stop')
        break
time2 = time.time()
# Save total and per-band train/validation histories using the original filenames.
lossroot = os.path.join(root_, "loss")
if not os.path.exists(lossroot):
    os.makedirs(lossroot)
with open(os.path.join(lossroot, "TrainLoss.txt"), 'w') as tl:
    for value in train_losses:
        tl.write(str(np.array(value)) + '\n')

with open(os.path.join(lossroot, "TrainLossLow.txt"), 'w') as tlLow:
    for value in train_losses_LowFre:
        tlLow.write(str(np.array(value)) + '\n')

with open(os.path.join(lossroot, "TrainLossMiddle.txt"), 'w') as tlMiddle:
    for value in train_losses_MiddleFre:
        tlMiddle.write(str(np.array(value)) + '\n')

with open(os.path.join(lossroot, "TrainLossHigh.txt"), 'w') as tlHigh:
    for value in train_losses_HighFre:
        tlHigh.write(str(np.array(value)) + '\n')

with open(os.path.join(lossroot, "ValidationLossLow.txt"), 'w') as vlLow:
    for value in validation_losses_LowFre:
        vlLow.write(str(np.array(value)) + '\n')

with open(os.path.join(lossroot, "ValidationLossMiddle.txt"), 'w') as vlMiddle:
    for value in validation_losses_MiddleFre:
        vlMiddle.write(str(np.array(value)) + '\n')

with open(os.path.join(lossroot, "ValidationLossHigh.txt"), 'w') as vlHigh:
    for value in validation_losses_HighFre:
        vlHigh.write(str(np.array(value)) + '\n')

with open(os.path.join(lossroot, "ValidationLoss.txt"), 'w') as vl:
    for value in validation_losses:
        vl.write(str(np.array(value)) + '\n')

# Record the loop duration in seconds, with the same timing scope described above.
time = time2-time1
with open(os.path.join(lossroot, "trainingtime.txt"), 'w') as tt:
    tt.write(str(time))
# Plot the recorded epoch means; this display operation does not affect training.
plot_path = os.path.join(lossroot, "learning_curve.jpg")
plt.figure(figsize=(8, 8))
plt.plot(train_losses)
plt.plot(validation_losses)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.title("Learning curve", fontsize=20)
plt.legend(['training loss', 'validation loss'], fontsize=15, loc='upper right')
plt.savefig(plot_path)
