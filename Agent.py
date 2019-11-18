import neat
from Environment import drive
import gym
from gym import wrappers, logger
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

env = gym.make('CarRacing-v0')
observation_space = env.reset()
h = open('results_with_1000_hiddennodes.txt', 'w+')
class CarDrivers:

    def evaluate_genomes(self, genomes, config):
        convolutional_net = load_model('convolutional_network_model_smaller_outputsize')
        for genome_id, genome in genomes:
            genome.fitness = 0
            observation = env.reset()
            observation = observation.reshape([1, 96, 96, 3]) / 255
            net = neat.nn.RecurrentNetwork.create(genome, config)

            for _ in range(500):
                env.render()
                observation = convolutional_net.predict(observation).flatten()

                action = net.activate(observation)
                action[0] = (((action[0] - 0) * (1 - (-1))/(1-0)) + (-1))

                observation, reward, done, info = env.step(action)
                observation = observation.reshape([1, 96, 96, 3]) / 255
                genome.fitness += reward

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
        population.add_reporter(neat.Checkpointer(5))

        winner = population.run(self.evaluate_genomes, 100)

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        print(winner_net)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    driver = CarDrivers()
    driver.run('config')
    h.close()

