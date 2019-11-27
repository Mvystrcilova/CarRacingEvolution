import neat
# from Environment import drive
import gym, numpy
# from keras.models import load_model
from PIL import Image


env = gym.make('CarRacing-v0')
observation_space = env.reset()
h = open('results_checkpoint_3.txt', 'w+')
class CarDrivers:

    def evaluate_genomes(self, genomes, config):
        # convolutional_net = load_model('cnn_rgb_scaled_36x60/cnn_rgb_scaled_36x60_model')
        for genome_id, genome in genomes:
            genome.fitness = 0

            observation = env.reset()
            observation = observation.reshape([400, 600, 3])
            observation = resize_image(36, 60, observation)

            net = neat.nn.RecurrentNetwork.create(genome, config)

            for _ in range(1000):
                env.render()
                # observation = convolutional_net.predict(observation).flatten()
                observation = observation.reshape([6480])

                action = net.activate(observation)
                action[0] = (((action[0] - 0) * (1 - (-1))/(1-0)) + (-1))
                # print(action[0], action[1], action[2])

                observation, reward, done, info = env.step(action)
                observation = observation.reshape([400, 600, 3])
                observation = resize_image(36, 60, observation)
                genome.fitness += reward

                if done:
                    print('done')
                    env.render()

            message = 'genome fitness: ' + str(genome.fitness) + ' of genome with id: ' + str(genome_id) + '\n'
            h.write(message)


    def run(self, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.Checkpointer(10, filename_prefix='neat_checkpoints/checkpoint_2_'))

        winner = population.run(self.evaluate_genomes, 300)

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        print(winner_net)

def resize_image(width, heigth, array):
    array = array[:350][:][:]
    img = Image.fromarray(array, 'RGB')
    img1A = img.resize((width, heigth), Image.ANTIALIAS)
    pix = numpy.array(img1A)
    pix = pix.reshape([1, width, heigth, 3]) / 255
    return pix


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    driver = CarDrivers()
    driver.run('config')
    h.close()

