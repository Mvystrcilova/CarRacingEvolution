import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import numpy, os, pickle, re, glob
# from PIL import Image
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def process_image(observation):
    observation = observation.reshape([96, 96, 3])
    observation = observation.astype('float32')
    return observation


def create_dataset():
    training_images = numpy.zeros([23126, 350, 600, 3])
    i = 0
    for filename in os.listdir('mnt/0/rgb_observations'):
        observation = numpy.load('mnt/0/rgb_observations/' + filename)
        # observation = process_image(observation)
        observation = observation / 255
        training_images[i] = observation.reshape([400, 600, 3])[:350][:][:]
        i += 1
        print(i)
    print(training_images.size)
    numpy.save('mnt/0/rgb_observation_file', training_images)




def train_network(input_file):
    input_image = Input(shape=(96, 96, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((3, 3), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((3,3), padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((3, 3))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_image, decoded)
    autoencoder.summary()
    encoder = Model(input_image, encoded)
    encoder.summary()

    autoencoder.compile(optimizer='adam', loss='mse')
    input_data = numpy.load(input_file)
    input_data = input_data
    hist = autoencoder.fit(input_data, input_data, epochs=70, batch_size=256, verbose=True)

    with open('histories/convolutional_network_training_history', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

    autoencoder.save('convolutional_network_autoencoder')
    encoder.save('convolutional_network_model_oneInLastDimension_28K_6x6')

def generate_input(spec_directory, batch_size, scale):
    while True:
        specs = []
        i = 0
        files = sorted(glob.glob(spec_directory + '/*.npy'), key=numericalSort)

        while len(specs) < batch_size:
            if (i > 23126):
                i = 0
            spec = numpy.load(files[i]).reshape([400,600,3])
            spec = spec[:350][:][:]
            # print(files[i])
            # if (i % 1000) == 0:
            #     img = Image.fromarray(spec, 'RGB')
            #     img.show()
            if scale:
                spec = spec/255
            specs.append(spec)
            i = i + 1
        yield ((numpy.array(specs)), numpy.array(specs))

def train_unscaled_rgb_network(input_file):
    input_image = Input(shape=(350, 600, 3))
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((5, 5), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((4, 4), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((5, 5))(x)
    decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_image, decoded)
    autoencoder.summary()
    encoder = Model(input_image, encoded)
    encoder.summary()
    # input = numpy.load(input_file)
    checkpoint = ModelCheckpoint('mnt/0/unscaled_cnn_autoencoder_rgb', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    adam = Adam(learning_rate=0.00001)
    autoencoder.compile(optimizer=adam, loss='mse')
    trainGen = generate_input(spec_directory='mnt/0/rgb_observations', batch_size=64, scale=False)
    callbacklist = [checkpoint]
    # hist = autoencoder.fit(input, input, batch_size=128, epochs=60, verbose=True)
    hist = autoencoder.fit_generator(trainGen, epochs=20, steps_per_epoch=360, verbose=True, callbacks=callbacklist)

    # encoder.save('/mnt/0/convolutional_network_model_rgb')
    #
    with open('mnt/0/histories/unscaled_cnn_rgb_training_history', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)


def train_rgb_network_scaled_36x60(input_file, history_file):
    input_image = Input(shape=(36, 60, 3))
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((3, 3), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((3, 3))(x)
    decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_image, decoded)
    autoencoder.summary()
    encoder = Model(input_image, encoded)
    encoder.summary()
    input_array = numpy.load(input_file)
    input_array = input_array/255
    checkpoint = ModelCheckpoint('mnt/0/cnn_rgb_scaled_36x60/cnn_rgb_scaled_36x60_model', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    adam = Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    callbacklist = [checkpoint]
    hist = autoencoder.fit(input_array, input_array, epochs=120, batch_size=256, verbose=True, callbacks=callbacklist)

    with open(history_file, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)


def train_rgb_network_scaled_72x120(input_file, history_file):
    input_image = Input(shape=(72, 120, 3))
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_image, decoded)
    autoencoder.summary()
    encoder = Model(input_image, encoded)
    encoder.summary()
    input_array = numpy.load(input_file)
    input_array = input_array/255
    checkpoint = ModelCheckpoint('mnt/0/cnn_rgb_scaled_72x120/cnn_rgb_scaled_72x120_model', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    adam = Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    callbacklist = [checkpoint]
    hist = autoencoder.fit(input_array, input_array, epochs=120, batch_size=64, verbose=True, callbacks=callbacklist)

    with open(history_file, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

def train_rgb_network(input_file):
    input_image = Input(shape=(350, 600, 3))
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((5, 5), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((4, 4), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((5, 5))(x)
    decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_image, decoded)
    autoencoder.summary()
    encoder = Model(input_image, encoded)
    encoder.summary()
    # input = numpy.load(input_file)
    checkpoint = ModelCheckpoint('mnt/0/cnn_autoencoder_rgb_lr5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    adam = Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    trainGen = generate_input(spec_directory='mnt/0/rgb_observations', batch_size=64, scale=True)
    callbacklist = [checkpoint]
    # hist = autoencoder.fit(input, input, batch_size=128, epochs=60, verbose=True)
    hist = autoencoder.fit_generator(trainGen, epochs=20, steps_per_epoch=360, verbose=True, callbacks=callbacklist)

    # encoder.save('/mnt/0/convolutional_network_model_rgb')
    #
    with open('mnt/0/histories/cnn_rgb_training_history_lr5', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    #
    # second_model = load_model('mnt/0/convolutional_network_autoencoder_rgb')
    # checkpoint = ModelCheckpoint('mnt/0/convolutional_network_autoencoder_rgb', monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacklist = [checkpoint]
    # hist2 = second_model.fit_generator(trainGen, epochs=10, steps_per_epoch=360, verbose=True, callbacks=callbacklist)


    # with open('mnt/0/histories/convolutional_network_training_history_2', 'wb') as file_pi:
    #     pickle.dump(hist2.history, file_pi)

    autoencoder.save('mnt/0/cnn_autoencoder_2_lr5_rgb')
    encoder.save('/mnt/0/cnn_encoder_2_lr5_rgb')

def train_again(model_file, input_file):
    model = load_model(model_file)
    input_arr = numpy.load(input_file)
    input_arr = input_arr/255
    # encoder = K.function([model.layers[0].input], model.layers[5])
    # print(encoder.outputs.shape)
    checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacklist = [checkpoint]
    hist = model.fit(input_arr, input_arr, epochs=120,batch_size=64, verbose=True, callbacks=callbacklist)

    # encoder.save('/mnt/0/convolutional_network_encoder_rgb')

    with open('mnt/0/cnn_rgb_scaled_72x120/cnn_rgb_scaled_72x120_model_history_2', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

# def resize_images():
#     i = 0
#     images_72x120 = numpy.zeros([23125, 72, 120, 3])
#
#     for filename in os.listdir('/Users/m_vys/Downloads/rgb_observations'):
#         array = numpy.load('/Users/m_vys/Downloads/rgb_observations/'+ filename)
#
#         array = array.reshape([400, 600, 3])
#         array = array[:350][:][:]
#         img = Image.fromarray(array, 'RGB')
#
#         h1 = 36
#         w1 = 60
#
#         h2 = 72
#         w2 = 120
#
#         h3 = 175
#         w3 = 300
#
#         img1A = img.resize((w2, h2), Image.ANTIALIAS)
#         pix = numpy.array(img1A)
#         images_72x120[i] = pix
#         print(i)
#         i += 1
#
#     numpy.save('images_72x120', images_72x120)


        # show_image(w1, h1, img)
        # show_image(w2, h2, img)
        # show_image(w3, h3, img)

# def show_image(width, height, img):
#     img1N = img.resize((width, height), Image.NEAREST)
#     img1B = img.resize((width, height), Image.BILINEAR)
#     img1BC = img.resize((width, height), Image.BICUBIC)
#     img1A = img.resize((width, height), Image.ANTIALIAS)
#
#     img1N.show()
#     img1B.show()
#     img1BC.show()
#     img1A.show()
#
#     return img1A


# create_dataset()
# train_rgb_network('/Users/m_vys/Downloads/rgb_observation_file.npy')
# train_again('mnt/0/cnn_rgb_scaled_36x60/cnn_rgb_scaled_36x60_model', 'mnt/0/cnn_rgb_scaled_36x60/images_36x60.npy')
train_again('mnt/0/cnn_rgb_scaled_72x120/cnn_rgb_scaled_72x120_model', 'mnt/0/cnn_rgb_scaled_72x120/images_72x120.npy')

# resize_images()
# train_unscaled_rgb_network('bla bla')
# train_rgb_network_scaled_36x60('mnt/0/cnn_rgb_scaled_36x60/images_36x60.npy', 'mnt/0/cnn_rgb_scaled_36x60/cnn_rgb_scaled_36x60_model_history_1')
# train_rgb_network_scaled_72x120('/mnt/0/cnn_rgb_scaled_72x120/images_72x120.npy', 'mnt/0/cnn_rgb_scaled_72x120/cnn_rgb_scaled_72x120_model_history_1')