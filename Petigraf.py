import math, pygame, sys
from re import L
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm
import os


####The game

done = 0

class Environment ():


    def __init__(self):
        self.moveReward = 1
        self.endReward = 50
        self.wrongPunishment = 999
        self.WIDTH = 500
        self.HEIGHT = 500

        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Vonalazzá")
        pygame.init()

        self.WHITE = (255,255,255)
        self.BLACK = (0,0,0)
        self.BLUE = (0,0,255)
        self.RED = (255,0,0)
        self.FONT_IMPACT = pygame.font.SysFont("impact", 50)

        self.FPS = 60

        self.turns = ["", self.RED, self.BLUE]
        self.vertices = [{"pos":(50,250),"color":self.BLACK, "key":0},{"pos":(125,50),"color":self.BLACK, "key":1},{"pos":(375,50),"color":self.BLACK, "key":2},{"pos":(450,250),"color":self.BLACK, "key":3},{"pos":(375,450),"color":self.BLACK, "key":4},{"pos":(125,450),"color":self.BLACK, "key":5}]

        self.turn= 1
        self.chosen= ""
        self.missing = []
        self.allLines = []
        self.lines = []
        self.possibleLines = []

        for i in range(len(self.vertices)-1):
            for j in range(i+1, len(self.vertices)):
                self.allLines.append((i, j))
                self.possibleLines.append((i, j))
                self.lines.append(0)
    
        
    
    def reset(self):

        self.turn= 1
        self.chosen= ""
        self.missing = []
        self.allLines = []
        self.lines = []
        self.possibleLines = []

        for i in range(len(self.vertices)-1):
            for j in range(i+1, len(self.vertices)):
                self.allLines.append((i, j))
                self.possibleLines.append((i, j))
                self.lines.append(0)

        return self.lines


    def drawLine(self, start, end):
        global done
        if(self.lines[self.allLines.index((start, end))] == 0):
            for i in range(len(self.lines)):
                if(self.lines[i] == self.turn):
                    if start == self.allLines[i][0]:
                        self.missing.append((self.turn, min(end, self.allLines[i][1]), max(end, self.allLines[i][1])))
                    if end == self.allLines[i][0]:
                        self.missing.append((self.turn, min(start, self.allLines[i][1]), max(start, self.allLines[i][1])))
                    if start == self.allLines[i][1]:
                        self.missing.append((self.turn, min(end, self.allLines[i][0]), max(end, self.allLines[i][0])))
                    if end == self.allLines[i][1]:
                        self.missing.append((self.turn, min(start, self.allLines[i][0]), max(start, self.allLines[i][0])))
            self.lines[self.allLines.index((start, end))] = self.turn
            self.possibleLines.remove((start, end))
            if (-self.turn, start, end) in self.missing:
                self.missing.remove((-self.turn, start, end))
            if (self.turn, start , end) in self.missing :
                # self.draw_window()
                # self.gameover()
                done = self.turn
            else:
                self.turn = -self.turn
            reward = 0
        else:
            reward = self.wrongPunishment
        return self.lines, reward, done 
    
    def action(self, choice):
        self.drawLine(self.allLines[choice][0], self.allLines[choice][1])


    ####Pygame-related code

    def draw_window(self):
        self.WIN.fill(self.WHITE)
        for vertex in self.vertices:
            pygame.draw.circle(self.WIN,vertex["color"],vertex["pos"],20)
        for i in range(len(self.lines)):
            if self.lines[i] != 0:
                pygame.draw.line(self.WIN, self.turns[self.lines[i]], self.vertices[self.allLines[i][0]]["pos"], self.vertices[self.allLines[i][1]]["pos"], width=5)
        pygame.display.update()

    def setup(self):
        pass

    def gameover(self):
        self.draw_window()
        pygame.time.delay(1000)
        pygame.quit()
        sys.exit()

    def main(self):
        clock=pygame.time.Clock()
        run = True
        while run:
            clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for vertex in self.vertices:
                            if event.pos[0] > vertex["pos"][0] - 20 and event.pos[0] < vertex["pos"][0] + 20 and event.pos[1] > vertex["pos"][1] - 20 and event.pos[1] < vertex["pos"][1] + 20:
                                if type(self.chosen) == str:
                                    self.chosen = vertex
                                    vertex["color"] = self.turns[self.turn]
                                else:
                                    if vertex["color"] != self.BLACK:
                                        self.chosen = ""
                                        vertex["color"] = self.BLACK
                                    elif self.lines[self.allLines.index((min(self.chosen["key"], vertex["key"]), max(self.chosen["key"], vertex["key"])))] != 0:
                                        pass
                                    else:
                                        self.drawLine(min(vertex["key"], self.chosen["key"]), max(vertex["key"], self.chosen["key"]))
                                        vertex["color"] = self.BLACK
                                        self.vertices[self.chosen["key"]]["color"] = self.BLACK
                                        self.chosen = ""

            self.draw_window()

        pygame.quit()
        sys.exit()


game = Environment()

# if __name__ == "__main__":
#     game.main()


if not os.path.isdir('models'):
    os.makedirs('models')


#### DQN stuff

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "valami"
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key,value,step=self.step)
                self.writer.flush()

class Agent():

    

    def __init__(self):

        self.ACTION_SPACE_SIZE = len(game.lines)-1 

        self.model = self.createModel()
        
        self.clearMemory()

        self.tensorboard = ModifiedTensorBoard  (log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        

    def clearMemory(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.info = []

    def addToMemory(self, newObservation, newAction, newReward):
        print(newObservation, newAction, newReward)
        self.observations.append(newObservation)
        self.actions.append(newAction)
        self.rewards.append(newReward)


    def createModel(self):
        model = Sequential()

        model.add(Flatten())

        model.add(Dense(50, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(50, activation="relu"))

        model.add(Dense(self.ACTION_SPACE_SIZE))

        return model

    
    def computeLoss(self, logits, actions, rewards):
        negLogprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
        loss = tf.reduce_mean(negLogprob*rewards)
        return loss
    
    def train(self, observations, actions, rewards):
        with tf.GradientTape() as tape:
            logits = self.model(observations)
            loss = self.computeLoss(logits, actions, rewards)
            grads = tape.gradient(loss, self.model.trainable_variables)

            Adam.apply_gradients(zip(grads, self.model.trainable_variables))

    def getAction(self, observation, epsilon):

        act = np.random.choice(['model', 'random'], 1, p=[1-epsilon, epsilon])[0]
        logits = self.model.predict(observation)
        probWeights = tf.nn.softmax(logits)[0]

        if act == 'model':

            sortedWeights = probWeights.sort()

            if(game.turn == -1):
                sortedWeights.reverse()
            
            choice = 0

            while game.lines[probWeights.index(sortedWeights[choice])] != 0:
                choice += 1
            
            action = probWeights.index(sortedWeights[choice])
        
        else:
            action = np.random.randint(0, len(game.lines))
            while game.lines[action] != 0:
                action = np.random.randint(0, len(game.lines))
        
        return action, probWeights
    

        


        
        
    


agent = Agent()

EPISODES = 20000
AGGREGATE_STATS_EVERY = 500

epsilon = 1
epsilonDecay = 0.9995
episodeRewards = []

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episodes"):
    agent.tensorboard.step = episode

    episodeReward = 0

    steps = 1

    currentState = game.reset()
    agent.clearMemory()

    done = 0

    while done == 0:
        action, probWeights = agent.getAction(currentState, epsilon)

        newState, reward, done = game.drawLine(game.allLines[action][0], game.allLines[action][1])

        currentState = newState
        steps += 1
        
        if done != 0:
            reward += -done * game.endReward - (15 - len(game.possibleLines))
    
        agent.addToMemory(newState, action, reward)

        episodeReward += reward
    
    
    agent.train(observations=np.array(agent.observations), actions=np.array(agent.actions), rewards=agent.rewards)

        
    episodeRewards.append(episodeReward)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        avg_reward = sum(episodeRewards[-AGGREGATE_STATS_EVERY:])/len(episodeRewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(episodeRewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(episodeRewards[-AGGREGATE_STATS_EVERY:])

        agent.tensorboard.update_stats(reward_avg = avg_reward, reward_min = min_reward, reward_max = max_reward, epsilon=epsilon)

        agent.model.save(f'models/{MODEL_NAME}1__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


    epsilon *= epsilonDecay    