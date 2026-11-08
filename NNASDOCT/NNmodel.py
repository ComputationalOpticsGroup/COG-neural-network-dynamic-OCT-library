"""Proposed dual-input, 16-frame AS-DOCT model"""

# Tensor layout throughout the encoder/decoder: (B, T, H, W, C).
# B = batch, T = 16 sequence points, H = depth, W = fast-scan position.
# The two inputs are OCT amplitude and its full FFT magnitude, respectively.
# Only the OCT branch has a physical time axis; the spectrum branch uses FFT-index order.
import tensorflow as tf
import tensorflow.keras as keras


# Build the two-input U-Net and return three single-channel AS-DOCT predictions.
# kernel1 controls Conv3D; kernel2 controls the spatial kernels in ConvLSTM2D.
def LSTM_3dUnet_structure(input1, input2,kernel1, kernel2):

    # Pool only the spatial dimensions; keep all 16 sequence points.
    # Max and average pooling both halve H and W; neither pools the sequence axis.
    # Concatenation gives (B, T, H/2, W/2, 2*C), rather than averaging the two results.
    def doublepool3d(input):
        x1 = keras.layers.MaxPool3D(pool_size=(1, 2, 2), padding='valid')(input)
        x2 = keras.layers.AvgPool3D(pool_size=(1, 2, 2), padding='valid')(input)
        return tf.concat([x1, x2], axis=-1)
    # Preserve the original sequence-axis normalization (axis=-4).
    # For a five-dimensional tensor, axis=-4 selects T, not the feature channels.
    # center=False disables the learned offset; scale=True keeps a learned scale.
    def layernorm(x):
        norm = keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=True, axis=-4)(x)
        return norm

    # x is the encoder skip feature; y is the lower-resolution decoder feature.
    # The transpose convolution doubles H/W, preserves T, and produces 'filter' channels.
    # Concatenate the upsampled feature with the matching-resolution skip feature.
    def up(x, y, filter):
        dconv = tf.keras.layers.Conv3DTranspose(filter, (1, 2, 2), strides=(1, 2, 2))
        y1 = dconv(y)
        xy = tf.concat([x, y1], axis=-1)
        return xy

    # Extract local sequence/spatial features with Conv3D, then apply layer normalization.
    # kernel=2 means a 2 x 2 x 2 convolution; padding='same' preserves T, H, and W.
    def conv3d_relu(input, filter, kernel):
        conv1 = tf.keras.layers.Conv3D(filter, kernel_size=kernel, padding='same', activation="relu")
        x1 = conv1(input)
        x1 = layernorm(x1)
        return x1

    # Separate OCT-amplitude and pseudo-amplitude-spectrum input branches.
    # OCT branch: keep full spatial resolution, with 64 output feature channels.
    # return_sequences=True retains all 16 sequence positions for later stages.
    x1 = conv3d_relu(input1, 64, kernel1)
    x1 = keras.layers.ConvLSTM2D(filters=64, kernel_size=kernel2, padding='same', return_sequences=True)(x1)

    # Spectrum branch: use the same layer structure with independently learned weights.
    # ConvLSTM processes the stored FFT-magnitude points; these are not temporal samples.
    a1 = conv3d_relu(input2, 64, kernel1)
    a1 = keras.layers.ConvLSTM2D(filters=64, kernel_size=kernel2, padding='same', return_sequences=True)(a1)

    # Fuse the two 64-channel branches into 128 channels, then map them back to 64.
    # This fused feature x1 is also the skip connection at the original spatial resolution.
    x1 = tf.concat([x1, a1], axis=-1)
    x1 = conv3d_relu(x1, 64, kernel1)
    x1 = keras.layers.ConvLSTM2D(filters=64, kernel_size=kernel2, padding='same', return_sequences=True)(x1)

    # Encoder: four spatial pooling stages.
    # After this stage: x2 has shape (B, T, H/2, W/2, 128).
    x2 = doublepool3d(x1)
    x2 = conv3d_relu(x2, 128, kernel1)
    x2 = keras.layers.ConvLSTM2D(128, kernel2, padding='same', return_sequences=True)(x2)

    # After this stage: x3 has shape (B, T, H/4, W/4, 256).
    x3 = doublepool3d(x2)
    x3 = conv3d_relu(x3, 256, kernel1)
    x3 = keras.layers.ConvLSTM2D(256, kernel2, padding='same', return_sequences=True)(x3)

    # After this stage: x4 has shape (B, T, H/8, W/8, 512).
    x4 = doublepool3d(x3)
    x4 = conv3d_relu(x4, 512, kernel1)
    x4 = keras.layers.ConvLSTM2D(512, kernel2, padding='same', return_sequences=True)(x4)

    # Bottleneck: x5 has shape (B, T, H/16, W/16, 1024).
    # For a 64 x 64 training patch, this stage has 4 x 4 spatial pixels.
    x5 = doublepool3d(x4)
    x5 = conv3d_relu(x5, 1024, kernel1)
    x5 = keras.layers.ConvLSTM2D(1024, kernel2, padding='same', return_sequences=True)(x5)

    # Decoder: upsampling with skip connections.
    # Restore H/8 x W/8 resolution and combine with x4; output channels are 512.
    y4 = up(x4, x5, 512)
    y4 = conv3d_relu(y4, 512, kernel1)
    y4 = keras.layers.ConvLSTM2D(512, kernel2, padding='same', return_sequences=True)(y4)

    # Restore H/4 x W/4 resolution and combine with x3; output channels are 256.
    y3 = up(x3, y4, 256)
    y3 = conv3d_relu(y3, 256, kernel1)
    y3 = keras.layers.ConvLSTM2D(256, kernel2, padding='same', return_sequences=True)(y3)

    # Restore H/2 x W/2 resolution and combine with x2; output channels are 128.
    y2 = up(x2, y3, 128)
    y2 = conv3d_relu(y2, 128, kernel1)
    y2 = keras.layers.ConvLSTM2D(128, kernel2, padding='same', return_sequences=True)(y2)

    # Output order is low, middle, high, as assigned by the training script.
    # The legacy r/g/b names do not indicate RGB display-channel order.
    # Low-frequency head: restore H x W and reduce features to 16 channels.
    # The final ConvLSTM uses its default return_sequences=False: its final hidden state
    # has shape (B, H, W, 16). Conv2D maps it to (B, H, W, 1), with nonnegative output.
    # Training targets are log10 AS amplitudes, so predictions use that same value scale.
    r1 = up(x1, y2, 64)
    r1 = conv3d_relu(r1, 16, kernel1)
    r1 = keras.layers.ConvLSTM2D(16, kernel2, padding='same')(r1)
    r0 = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(r1)

    # Middle-frequency head: the same operations, with a separate set of learned weights.
    g1 = up(x1, y2, 64)
    g1 = conv3d_relu(g1, 16, kernel1)
    g1 = keras.layers.ConvLSTM2D(16, kernel2, padding='same')(g1)
    g0 = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(g1)

    # High-frequency head: another independent output head at full spatial resolution.
    b1 = up(x1, y2, 64)
    b1 = conv3d_relu(b1, 16, kernel1)
    b1 = keras.layers.ConvLSTM2D(16, kernel2, padding='same')(b1)
    b0 = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(b1)

    # Keep this low/middle/high order consistent with AS targets in NNtrain.py.
    return r0, g0, b0

# Public model constructor used by NNtrain.py.
# None permits different spatial sizes; four pooling stages require multiples of 16.
def LSTM_3dUnet():
    # Inputs: (batch, 16, depth, fast-scan position, 1).
    # Spatial dimensions must be divisible by 16 (training patches: 64 x 64).
    Input1 = tf.keras.layers.Input(shape=[16, None, None, 1])
    Input2 = tf.keras.layers.Input(shape=[16, None, None, 1])
    lowfre, middlefre, highfre = LSTM_3dUnet_structure(Input1, Input2, kernel1=2, kernel2=2)
    # Model input order: [OCT amplitude, pseudo-amplitude spectrum].
    # Model output order: [low frequency, middle frequency, high frequency].
    return tf.keras.Model(inputs=[Input1, Input2], outputs=[lowfre, middlefre, highfre])
