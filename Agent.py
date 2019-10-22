import neat
from Environment import drive
import gym
from gym import wrappers, logger

env = gym.make('CarRacing-v0')
observation_space = env.reset()
h = open('results', 'w+')
class CarDrivers:

    def evaluate_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0
            observation = env.reset()
            observation = observation.reshape([27648, 1])
            net = neat.nn.RecurrentNetwork.create(genome, config)
            # net_observation, net_reward = drive(net)
            for _ in range(500):
                env.render()
                # print(env.action_space.shape)
                # x = env.observation_space
                action = net.activate(observation)
                # action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                observation = observation.reshape([27648, 1])
                # print(reward, _)
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

        winner = population.run(self.evaluate_genomes, 300)

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        print(winner_net)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    driver = CarDrivers()
    driver.run('config')
    h.close()

