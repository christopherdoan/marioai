import retro
import numpy as np
import cv2
import pickle
import neat
import gym

# class Discretizer(gym.ActionWrapper):
#     """
#     Wrap a gym environment and make it use discrete actions.
#     Args:
#         combos: ordered list of lists of valid button combinations
#     """

#     def __init__(self, env, combos):
#         super().__init__(env)
#         assert isinstance(env.action_space, gym.spaces.MultiBinary)
#         buttons = env.unwrapped.buttons
#         self._decode_discrete_action = []
#         for combo in combos:
#             arr = np.array([False] * env.action_space.n)
#             for button in combo:
#                 arr[buttons.index(button)] = True
#             self._decode_discrete_action.append(arr)

#         self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

#     def action(self, act):
#         return self._decode_discrete_action[act].copy()
    
# #BASELINE TO CREATE WRAPPER FOR ROBOTNIK
# class RobotnikDiscretizer(Discretizer):
#     """
#     Use Sonic-specific discrete actions
#     based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
#     """
#     def __init__(self, env):
#         super().__init__(env=env, combos=[['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']])





#Create gym-retro environment for game of choice: DRMBM

imgarray = []

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = retro.make(game = 'SuperMarioBros3-NES')
        # Set observation variable (Age)
        self.env.reset()

        # Create a random action (generic)
        ac = self.env.action_space.sample()
        ob, _, _, _ = self.env.step(ac)

        # Get the x, y and colors of the input space (from the emulator)
        inx, iny, inc = self.env.observation_space.shape
        # Divide input by 8
        inx, iny, inc = int(inx/8), int(iny/8), int(inc/8)

        #Create NEAT net
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = -1
        xpos_max = 0
        done = False
        lives = -1

        while not done:
            #self.env.render()
            frame+= 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            #cut image that NN sees in half
            # ob= ob[0:ob.shape[0], 0:int(ob.shape[1]/2)]

            imgarray = np.ndarray.flatten(ob)
            
            nnOutput = net.activate(imgarray)
            
            #increment emulator by 1 step
            ob, rew, done, info = self.env.step(nnOutput)

            if lives == -1:
                lives = info['lives']
            if xpos == 0:
                    xpos = info['xpos']

            if xpos < info['xpos'] and info['xpos'] - xpos < 40000:
                fitness_current += (info['xpos'] - xpos)
                xpos = info['xpos']

            fitness_current += rew
        
            if fitness_current > current_max_fitness and lives == info['lives']:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            if done or counter == 150:
                done = True
            
            self.genome.fitness = fitness_current
        return current_max_fitness

def eval_genomes(genomes, config):
    worky = Worker(genomes, config)
    return worky.work()

if __name__ == "__main__":
    #Config File necessary to create a NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward.txt')

    #Create starting population
    p = neat.Population(config)



    #Compute statistic for the game
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(4, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

# # Below code creates random actions for the game
# obs = env.reset()
# while True:
#     obs, rew, done, info = env.step(env.action_space.sample())
#     print(rew)
#     env.render()
#     if done:
#         obs = env.reset()
# env.close()

