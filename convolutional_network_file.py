from keras.models import load_model
import numpy
from PIL import Image

def compare_images(cnn_network_file, image_file):
    model = load_model(cnn_network_file)

    img_array = numpy.load(image_file)
    img_array= img_array.reshape([400, 600, 3])
    img_array = img_array[:350][:][:]

    img_1 = Image.fromarray(img_array, 'RGB')
    img_1.show()

    img_array = img_array.reshape([1, 350, 600, 3]) / 255
    output = model.predict(img_array)
    output = output*255
    output = output.reshape([350,600,3])

    img_2 = Image.fromarray(output, 'RGB')
    img_2.show()

compare_images('convolutional_network_autoencoder_rgb_2', '/Users/m_vys/Downloads/rgb_observations/observations_2_2693.npy')