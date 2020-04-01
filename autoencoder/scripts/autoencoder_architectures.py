import setGPU

import tensorflow as tf

class ConvAutoEncoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.input_img = tf.keras.layers.Input(shape=self.input_shape, name='encoder_input')
        self.encoder = self.encoder(self.input_img)
        self.decoder = self.decoder(self.encoder)
        self.autoencoder = tf.keras.models.Model(self.input_img, self.decoder)
        #self.autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

    """ encoder """
    def encoder(self, input_img):
        filters = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 512]

        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #224 x 224 x 32
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) #112 x 112 x 32

        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #112 x 112 x 64
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2) #56 x 56 x 64

        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #56 x 56 x 128 (small and thick)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)

        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #56 x 56 x 256 (small and thick)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)

        conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = tf.keras.layers.BatchNormalization()(conv6)

        dense = tf.keras.layers.Dense(16, activation='relu', name='encoder')(conv6)

        return dense

    """ decoder """
    def decoder(self, dense):
        conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(dense) #56 x 56 x 128
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)

        conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5) #56 x 56 x 64
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
        conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = tf.keras.layers.BatchNormalization()(conv6)

        up1 = tf.keras.layers.UpSampling2D((2,2))(conv6) #112 x 112 x 64

        conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 112 x 112 x 32
        conv7 = tf.keras.layers.BatchNormalization()(conv7)
        conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = tf.keras.layers.BatchNormalization()(conv7)

        up2 = tf.keras.layers.UpSampling2D((2,2))(conv7) # 224 x 224 x 32

        conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
        conv8 = tf.keras.layers.BatchNormalization()(conv8)
        conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
        conv8 = tf.keras.layers.BatchNormalization()(conv8)

        up3 = tf.keras.layers.UpSampling2D((2, 2))(conv8)

        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
        conv9 = tf.keras.layers.BatchNormalization()(conv9)
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        conv9 = tf.keras.layers.BatchNormalization()(conv9)

        up4 = tf.keras.layers.UpSampling2D((2, 2))(conv9)

        decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up4) # 224 x 224 x 1

        return decoded

    """
    Typical CNN autoencoder architecture
    Todo:
    - feed more data
    - experiment with no of layers
    - experiment with no of neurons
    - experiment with different initializers
    - experiment with different activation functions
    - experiment with different optimizers
    - experiment with different loss functions
    """
    def ae_4layers(self):
        input_img = tf.keras.layers.Input(shape=(self.img_width, self.img_height, self.img_channels))

        """ Encoder layers """
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        #encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name='encoder')(x)

        """ Decoder layers """
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        ae = tf.keras.models.Model(input_img, decoded)

        return ae

""" VGG architecutre based autoencoder by initializing pre-trained networks """
class VGG16AutoEncoder:
    def __init__(self, input_shape, weight_file: str, num_of_layers_to_freeze_from_input_layer: int):
        self.encoder = VGG16Encoder(input_shape=input_shape, weight_file=weight_file, num_of_layers_to_freeze_from_input_layer=num_of_layers_to_freeze_from_input_layer).encoder
        self.decoder = Decoder(self.encoder).decoder
        self.auto_encoder = Model(inputs=self.encoder.inputs, outputs=self.decoder)
        #print(self.auto_encoder.summary())

class VGG16Encoder:
    def __init__(self, input_shape, weight_file: str, num_of_layers_to_freeze_from_input_layer: int):
        self.encoder = self.__get_encoder(input_shape)
        self.encoder.load_weights(weight_file)
        self.encoder = self.__freeze_weights(self.encoder, num_of_layers_to_freeze_from_input_layer)

    def __get_encoder(self, input_shape):
        return Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='encoder'),])

    def __freeze_weights(self, encoder, num_of_layers_to_freeze_from_input_layer: int):
        for i in range(num_of_layers_to_freeze_from_input_layer):
            encoder.layers[i].trainable= False

        return encoder

class Decoder:
    def __init__(self, encoder):
        self.decoder = self.__get_decoder(encoder)

    def __get_decoder(self, encoder):
        decoded_encoder = Dense(784, activation='relu')(encoder.output)
        decoded_encoder = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoded_encoder = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoded_encoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoded_encoder = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoded_encoder = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(decoded_encoder)
        decoded_encoder = UpSampling2D(size=(2, 2))(decoded_encoder)

        decoder = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(decoded_encoder)

        return decoder

"""
Auto encoder architecture built based on inception blocks
"""
class inceptionAutoencoder(object):
    def __init__(self, input_shape):
        self.input_img = tf.keras.layers.Input(shape=input_shape, name='encoder_input')

        self.encoder = InceptionEncoder(self.input_img).encoder
        #self.encoder_model = Model(inputs=self.input_img, outputs=self.encoder)
        self.decoder = InceptionDecoder(self.encoder).decoder

        self.autoencoder = tf.keras.models.Model(inputs=self.input_img, outputs=self.decoder)

"""
encoder architecture using inception blocks
"""
class InceptionEncoder(object):
    def __init__(self, input_img):
        self.kernel_init = tf.keras.initializers.glorot_uniform()
        self.bias_init = tf.keras.initializers.Constant(value=0.2)

        self.encoder = self.__get_encoder(input_img)

    """ Inception module block """
    def __inception_module(self, x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
        conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)

        conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_3x3)

        conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation = 'relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation = 'relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_5x5)

        pool_proj = tf.keras.layers.MaxPool2D((3, 3), strides = (1, 1), padding='same')(x)
        pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(pool_proj)

        output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

        return output

    """ Encoder architecture which is based on inception """
    def __get_encoder(self, input_img):
        #input_layer = Input(shape = input_shape)

        x = tf.keras.layers.Conv2D(64, (7,7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(input_img)
        x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)

        x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
        x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
        x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

        x = self.__inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32, name='inception_3a')

        x = self.__inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64 , name='inception_3b')
        x = tf.keras.layers.MaxPool2D((3,3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

        x = self.__inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64 , name='inception_4a')
        #x = AveragePooling2D((5, 5), strides=3)(x)
        x = tf.keras.layers.AveragePooling2D((2, 2), strides=2)(x)

        x = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu', name='encoder')(x)

        return x

"""
decoder architecture using inception blocks
"""
class InceptionDecoder(object):
    def __init__(self, encoder):
        self.kernel_init = tf.keras.initializers.glorot_uniform()
        self.bias_init = tf.keras.initializers.Constant(value=0.2)

        self.decoder = self.__get_decoder(encoder)

    """ Inception module block """
    def __inception_module(self, x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
        conv_1x1 = tf.keras.layers.Conv2DTranspose(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)

        conv_3x3 = tf.keras.layers.Conv2DTranspose(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        conv_3x3 = tf.keras.layers.Conv2DTranspose(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_3x3)

        conv_5x5 = tf.keras.layers.Conv2DTranspose(filters_5x5_reduce, (1, 1), padding='same', activation = 'relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        conv_5x5 = tf.keras.layers.Conv2DTranspose(filters_5x5, (5, 5), padding='same', activation = 'relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_5x5)

        #pool_proj = MaxPool2D((3, 3), strides = (1, 1), padding='same')(x)
        pool_proj = tf.keras.layers.Conv2DTranspose(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_5x5)

        output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

        return output

    def __get_decoder(self, encoder):
        x = tf.keras.layers.Conv2DTranspose(64, (7,7), padding='same', strides=(2, 2), activation='relu', name='convTran_1_7x7/2',
                kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(encoder)
        #x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)

        x = tf.keras.layers.Conv2DTranspose(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='convTran_2a_3x3/1')(x)
        x = tf.keras.layers.Conv2DTranspose(192, (3, 3), padding='same', strides=(2, 2), activation='relu', name='convTran_2b_3x3/1')(x)
        #x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

        x = self.__inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32, name='inceptionCT_3a')

        x = self.__inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64 , name='inceptionCT_3b')
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2), activation='relu', name='convTran_3b_3x3/1')(x)
        #x = MaxPool2D((3,3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

        x = self.__inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64 , name='inceptionCT_4a')
        #x = AveragePooling2D((5, 5), strides=3)(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2), activation='relu', name='convTran_4a_3x3/1')(x)

        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2), activation='relu', name='convTran_4a_1_3x3/1')(x)

        x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu')(x)

        return x

"""
Auto encoder architecture based on UNET
reference: http://deeplearning.net/tutorial/unet.html
"""
class UnetAutoEncoder:
    def __init__(self, input_shape):
        print("initialize")
        self.input_shape = input_shape

    def dice_coef(self, y_true, y_pred):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

    def dice_coef_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)

    """
    This architecture is more deeper and wider
    No.of units/nurons at the begging is low when compare to the lower layers, this means encoder feature vector size is more (8x8x512)
    Training is faster
    """
    def inference_UNET_1(self):
        inputs = tf.keras.layers.Input(self.input_shape, name='encoder_input')

        """ Encoder architecture """
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='encoder')(conv5)

        """ Decoder architecture """
        up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[conv10])
        model.summary()
        model.compile(optimizer='adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        return model

    """
    This architecutre is wider and deeper
    No.of units/nurons at the beginning layers are more compare to low layers of encoder, means feature vector size is lower (8x8x32)
    and trianing is ver slow as no.of parameters might be more
    """
    def inference_UNET_2(self):
        inputs = tf.keras.layers.Input(self.input_shape, name='encoder_input')

        """ Encoder layers """
        conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='encoder')(conv5)

        """ Decoder layers """
        up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[conv10])
        model.summary()
        model.compile(optimizer='adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        return model

    def inference_modified_UNET(self):
        inputs = tf.keras.layers.Input(self.input_shape, name='encoder_input')

        """ encoder layers """
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder')(conv5)

        """ decoder layers """
        up6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
        conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
        conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
        conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = tf.keras.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[conv10])
        model.summary()
        model.compile(optimizer='adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])

        return model

