import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from PIL import Image
import numpy, os, pickle, re, glob, sklearn

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
    training_images = numpy.zeros([23126, 600, 400, 3])
    i = 0
    for filename in os.listdir('/Users/m_vys/Downloads/rgb_observations'):
        observation = numpy.load('/Users/m_vys/Downloads/rgb_observations/' + filename)
        # observation = process_image(observation)
        observation = observation / 255
        training_images[i] = observation
        i += 1
        print(i)

    numpy.save('rgb_observation_file', training_images)




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

def generate_input(spec_directory, batch_size):
    while True:
        specs = []
        i = 0
        files = sorted(glob.glob(spec_directory + '/*.npy'), key=numericalSort)

        while len(specs) < batch_size:
            if (i > 23126):
                i = 0
            spec = numpy.load(files[i])
            # print(files[i])
            # if (i % 1000) == 0:
            #     img = Image.fromarray(spec, 'RGB')
            #     img.show()
            spec = spec/255
            specs.append(spec)
            i = i + 1
        yield ((numpy.array(specs)), numpy.array(specs))

def train_rgb_network(input_file):
    input_image = Input(shape=(600, 400, 3))
    x = Conv2D(32, (3, 2), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((3, 4), padding='same')(x)
    x = Conv2D(16, (3, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((5, 5), padding='same')(x)
    x = Conv2D(3, (5, 2), activation='relu', padding='same')(x)
    # x = MaxPooling2D((4, 2), padding='same')(x)
    # x = Conv2D(1, (3, 2), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((4, 4), padding='same')(x)
    x = UpSampling2D((3, 4))(x)
    x = Conv2D(16, (3, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((5, 5))(x)
    # x = Conv2D(12, (3, 2), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 4))(x)
    decoded = Conv2D(3, (3, 2), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_image, decoded)
    autoencoder.summary()
    encoder = Model(input_image, encoded)
    encoder.summary()

    autoencoder.compile(optimizer='adam', loss='mse')
    trainGen = generate_input(spec_directory='mnt/0/rgb_observations', batch_size=64)
    hist = autoencoder.fit_generator(trainGen, epochs=120, steps_per_epoch=360, verbose=True)

    with open('mnt/0/histories/convolutional_network_training_history', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

    autoencoder.save('mnt/0/convolutional_network_autoencoder_rgb')
    encoder.save('mnt/0/convolutional_network_model_rgb')

# create_dataset()
train_rgb_network('mnt/0/rgb_observation_file.npy')

