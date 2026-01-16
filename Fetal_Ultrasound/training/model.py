import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                                     concatenate, BatchNormalization, Dropout, 
                                     Activation, Add, Multiply, GlobalAveragePooling2D,
                                     Dense, Reshape, Lambda)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def conv_block(input_tensor, num_filters, dropout_rate=0.1):
    """
    Convolutional block with batch normalization and dropout.
    """
    x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def attention_block(x, g, inter_channel):
    """
    Attention gate for focusing on relevant features.
    
    Args:
        x: Features from encoder (skip connection)
        g: Features from decoder (gating signal)
        inter_channel: Number of intermediate channels
    """
    # Theta path (encoder features)
    theta_x = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(x)
    
    # Phi path (decoder features)
    phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(g)
    
    # Add and apply ReLU
    add_xg = Add()([theta_x, phi_g])
    add_xg = Activation('relu')(add_xg)
    
    # Psi path
    psi = Conv2D(1, (1, 1), padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    
    # Apply attention
    x = Multiply()([x, psi])
    
    return x

def residual_block(input_tensor, num_filters):
    """
    Residual block for better gradient flow.
    """
    x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    
    # Skip connection
    shortcut = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def squeeze_excite_block(input_tensor, ratio=8):
    """
    Squeeze and Excitation block for channel-wise attention.
    """
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se_shape = (1, 1, filters)
    
    # Squeeze: Global average pooling
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    
    # Excitation: FC -> ReLU -> FC -> Sigmoid
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    
    # Scale: Multiply input with attention weights
    x = Multiply()([input_tensor, se])
    
    return x

def AttentionUNet(input_size=(256, 256, 1), filters_base=32):
    """
    Attention U-Net with Squeeze-Excitation blocks for improved fetal ultrasound segmentation.
    
    Args:
        input_size: Input image size (height, width, channels)
        filters_base: Base number of filters (will be multiplied at each level)
    
    Returns:
        Keras Model
    """
    inputs = Input(input_size)
    
    # Encoder Path
    # Block 1
    c1 = conv_block(inputs, filters_base, dropout_rate=0.1)
    c1 = squeeze_excite_block(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    # Block 2
    c2 = conv_block(p1, filters_base * 2, dropout_rate=0.1)
    c2 = squeeze_excite_block(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Block 3
    c3 = conv_block(p2, filters_base * 4, dropout_rate=0.2)
    c3 = squeeze_excite_block(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Block 4
    c4 = conv_block(p3, filters_base * 8, dropout_rate=0.2)
    c4 = squeeze_excite_block(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck (Bridge)
    c5 = conv_block(p4, filters_base * 16, dropout_rate=0.3)
    c5 = squeeze_excite_block(c5)
    
    # Decoder Path with Attention Gates
    # Block 6
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(filters_base * 8, (2, 2), padding='same', kernel_initializer='he_normal')(u6)
    a6 = attention_block(c4, u6, filters_base * 4)
    u6 = concatenate([u6, a6])
    c6 = conv_block(u6, filters_base * 8, dropout_rate=0.2)
    
    # Block 7
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(filters_base * 4, (2, 2), padding='same', kernel_initializer='he_normal')(u7)
    a7 = attention_block(c3, u7, filters_base * 2)
    u7 = concatenate([u7, a7])
    c7 = conv_block(u7, filters_base * 4, dropout_rate=0.2)
    
    # Block 8
    u8 = UpSampling2D((2, 2))(c7)
    u8 = Conv2D(filters_base * 2, (2, 2), padding='same', kernel_initializer='he_normal')(u8)
    a8 = attention_block(c2, u8, filters_base)
    u8 = concatenate([u8, a8])
    c8 = conv_block(u8, filters_base * 2, dropout_rate=0.1)
    
    # Block 9
    u9 = UpSampling2D((2, 2))(c8)
    u9 = Conv2D(filters_base, (2, 2), padding='same', kernel_initializer='he_normal')(u9)
    a9 = attention_block(c1, u9, filters_base // 2)
    u9 = concatenate([u9, a9])
    c9 = conv_block(u9, filters_base, dropout_rate=0.1)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid', name='output')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs], name='AttentionUNet')
    
    return model

def UNet(input_size=(256, 256, 1)):
    """
    Standard U-Net architecture (fallback/baseline model).
    """
    inputs = Input(input_size)
    
    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bridge
    c5 = conv_block(p4, 512)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 256)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 128)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 64)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name='UNet')
    return model

if __name__ == "__main__":
    # Test model creation
    print("Creating Attention U-Net model...")
    model = AttentionUNet(input_size=(256, 256, 1), filters_base=32)
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
