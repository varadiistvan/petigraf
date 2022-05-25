import math
import pygame
import sys
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


# The game

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

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.FONT_IMPACT = pygame.font.SysFont("impact", 50)

        self.FPS = 60

        self.turns = ["", self.RED, self.BLUE]
        self.vertices = [{"pos": (50, 250), "color": self.BLACK, "key": 0}, {"pos": (125, 50), "color": self.BLACK, "key": 1}, {"pos": (375, 50), "color": self.BLACK, "key": 2}, {
            "pos": (450, 250), "color": self.BLACK, "key": 3}, {"pos": (375, 450), "color": self.BLACK, "key": 4}, {"pos": (125, 450), "color": self.BLACK, "key": 5}]

        self.turn = 1
        self.chosen = ""
        self.missing = []
        self.space = []

        for i in range(len(self.vertices)-1):
          self.space.append([])
          for j in range(len(self.vertices)-i-1):
            self.space[i].append(np.random.choice(3, 1)[0])

        print(self.space)
        print(self.isDone())
        
    
    def isDone(self, space = None):
      if(space == None):
        space = self.space

      for i in range(len(space)):
        for j in range(len(space[i])-1):
          if(space[i][j] != 0):
            for k in range(j+1, len(space[i])):
              if(space[i][j] == space[i][k] and space[i+j+1][k-j-1] == space[i][j]):
                return space[i][j]

      return 0

    def reset(self):
      pass

    def drawLine(self, start, end):
      pass

    def action(self, choice):
        start = round(0.0169*choice**2+0.0386*choice-0.0794) #excel
        end = round(0.5*start**2+0.0386*start)
        self.drawLine(start, end)

    # Pygame-related code

    def draw_window(self):
        self.WIN.fill(self.WHITE)
        for vertex in self.vertices:
            pygame.draw.circle(self.WIN, vertex["color"], vertex["pos"], 20)
        for i in range(len(self.space)):
            for j in range(len(self.space[i])):
                if self.space[i][j] != 0:
                    pygame.draw.line(self.WIN, self.turns[self.space[i][j]], self.vertices[i]["pos"], self.vertices[i+j+1]["pos"], width=5)
        pygame.display.update()

    def setup(self):
        pass

    def gameover(self):
        self.draw_window()
        pygame.time.delay(1000)
        pygame.quit()
        sys.exit()

    def main(self):
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for vertex in self.vertices:
                        if event.pos[0] > vertex["pos"][0] - 20 and event.pos[0] < vertex["pos"][0] + 20 and event.pos[1] > vertex["pos"][1] - 20 and event.pos[1] < vertex["pos"][1] + 20:
                          
                            if type(self.chosen) == str:
                                self.chosen = vertex
                                vertex["color"] = self.turns[self.turn]
                            else:
                                print(vertex["key"], self.chosen["key"])
                                if vertex["color"] != self.BLACK:
                                    self.chosen = ""
                                    vertex["color"] = self.BLACK
                                elif self.space[min(self.chosen["key"], vertex["key"]-1)][max(self.chosen["key"], vertex["key"])-min(self.chosen["key"], vertex["key"]-1)] != 0:
                                    pass
                                else:
                                    self.drawLine(min(vertex["key"], self.chosen["key"])-1, max(
                                        vertex["key"], self.chosen["key"])-1)
                                    vertex["color"] = self.BLACK
                                    self.vertices[self.chosen["key"]
                                                  ]["color"] = self.BLACK
                                    self.chosen = ""

            self.draw_window()

        pygame.quit()
        sys.exit()


game = Environment()

# if __name__ == "__main__":
#     game.main()

game.main()


if not os.path.isdir('models'):
    os.makedirs('models')
