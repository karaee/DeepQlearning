#!/usr/bin/env python
# coding: utf-8

# # Environment

# In[1]:


import pygame
import random
import time
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import math


# In[2]:


# window size
WIDTH = 1060
HEIGHT = 720
FPS = 30 # how fast game is

# colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255, 0, 0) # RGB
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255,255,0)

IMGHISTORY = 4
NBACTIONS = 4
NUMEPISODES = 2


# ## Player

# In[3]:


class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50,50))
        self.image.fill(WHITE)
     
        self.rect = self.image.get_rect()
        self.rect.centerx = 50
        self.rect.centery = HEIGHT/2
        
        self.x_speed = 0
        self.y_speed = 0
        
    def update(self,action):
        # Controls
        key_state = pygame.key.get_pressed()
                
        if key_state[pygame.K_w] or action == 0:
            self.y_speed = -15
            
        elif key_state[pygame.K_s] or action == 1:
            self.y_speed = 15
        
        elif key_state[pygame.K_a] or action == 2:
            self.x_speed = -15
            
        elif key_state[pygame.K_d] or action == 3:
            self.x_speed = 15
            
        
        #Control update 
        self.rect.x += self.x_speed
        self.rect.y += self.y_speed
        
        self.x_speed = 0
        self.y_speed = 0
        
        #Screeen options
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
            
        if self.rect.left < 0:
            self.rect.left = 0
        
        if self.rect.top < 0:
            self.rect.top = 0
        
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT


# ## Enemy

# In[4]:


class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((40,40))
        self.image.fill(RED)
      
        self.rect = self.image.get_rect()
        self.y_center = random.randint(15,HEIGHT-10)
        self.x_center = random.randint(WIDTH-400,WIDTH-20)
        self.rect.center = (self.x_center,self.y_center)
        
        self.x_speed = -10
        self.y_speed = 0
             
    def update(self):
        self.rect.x += self.x_speed
        
        if self.rect.left < 0:
            self.y_center = random.randint(15,HEIGHT-10)
            self.x_center = random.randint(WIDTH-400,WIDTH-20)
            self.rect.center = (self.x_center,self.y_center)


# ## Env

# In[5]:


class Env:
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.player_group = pygame.sprite.Group()
        self.enemy_group = pygame.sprite.Group()
        self.player = Player()
        self.player_group.add(self.player)
        #enemies
        self.enemy_1 = Enemy()
        self.enemy_group.add(self.enemy_1)
        self.enemy_2 = Enemy()
        self.enemy_group.add(self.enemy_2)
        self.enemy_3 = Enemy()
        self.enemy_group.add(self.enemy_3)
        self.enemy_4 = Enemy()
        self.enemy_group.add(self.enemy_4)        
        self.enemy_5 = Enemy()
        self.enemy_group.add(self.enemy_5)
        self.enemy_6 = Enemy()
        self.enemy_group.add(self.enemy_6)
        self.enemy_7 = Enemy()
        self.enemy_group.add(self.enemy_7)
        self.enemy_8 = Enemy()
        self.enemy_group.add(self.enemy_8)
        self.enemy_9 = Enemy()
        self.enemy_group.add(self.enemy_9)
        self.enemy_10 = Enemy()
        self.enemy_group.add(self.enemy_10)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()
    
    def step(self,action):#PlayNextMove
        
        #update
        self.player.update(action)
        self.enemy_group.update()
        
        ScreenImage = pygame.surfarray.array3d(pygame.display.get_surface())
        
        return ScreenImage 
    
    def initialStates(self): #reset()
        self.player_group = pygame.sprite.Group()
        self.enemy_group = pygame.sprite.Group()
        self.player = Player()
        self.player_group.add(self.player)
        #enemies
        self.enemy_1 = Enemy()
        self.enemy_group.add(self.enemy_1)
        self.enemy_2 = Enemy()
        self.enemy_group.add(self.enemy_2)
        self.enemy_3 = Enemy()
        self.enemy_group.add(self.enemy_3)
        self.enemy_4 = Enemy()
        self.enemy_group.add(self.enemy_4)        
        self.enemy_5 = Enemy()
        self.enemy_group.add(self.enemy_5)
        self.enemy_6 = Enemy()
        self.enemy_group.add(self.enemy_6)
        self.enemy_7 = Enemy()
        self.enemy_group.add(self.enemy_7)
        self.enemy_8 = Enemy()
        self.enemy_group.add(self.enemy_8)
        self.enemy_9 = Enemy()
        self.enemy_group.add(self.enemy_9)
        self.enemy_10 = Enemy()
        self.enemy_group.add(self.enemy_10)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        
        #state
        ScreenImage = pygame.surfarray.array3d(pygame.display.get_surface())
        return ScreenImage 
    
    def InitialDisplay(self):
        
        pygame.event.pump()
        
        #draw / render(show)
        screen.fill(BLACK)
        self.player_group.draw(screen)
        self.enemy_group.draw(screen)

        #after drawing flip display
        pygame.display.flip()
        
    
    def run(self):
        # Game Loop
        self.InitialDisplay()
        batch_size = 16
        running = True
        
        BestAction = 0
        InitialScreenImage = self.step(BestAction)
        InitialGameImage = ProcessGameImage(InitialScreenImage)
        
        GameState = np.stack((InitialGameImage,InitialGameImage,InitialGameImage,InitialGameImage),axis = 2)
        GameState = GameState.reshape(1, GameState.shape[0],GameState.shape[1],GameState.shape[2])
        
        while running:
            self.reward = 1
            done = False
            # keep loop running at the right speed
            clock.tick(FPS)
            
            # process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_q:
                        pygame.quit()
                                        
                    if event.key == pygame.K_p:
                        pause = 1
                        
                        while pause:
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    
                                    if event.key == pygame.K_p:
                                        pause = 0
                                        
                                    if event.key == pygame.K_q:
                                        pause = 0
                                        pygame.quit()
                                        return
                                        

            #update
            BestAction = self.agent.act(GameState)
            NewScreenImage = self.step(BestAction)
            
            NewGameImage = ProcessGameImage(NewScreenImage)
            NewGameImage = NewGameImage.reshape(1,NewGameImage.shape[0],NewGameImage.shape[1],1)
            
            NextState = np.append(NewGameImage, GameState[:,:,:,:3], axis = 3)
            self.agent.remember(GameState,BestAction,self.reward,NextState)
            
            self.total_reward += self.reward

            # check to see if a enemy hit the player
            hits = pygame.sprite.spritecollide(self.player, self.enemy_group, False)
            if hits: #hits == True
                self.reward = -400
                self.total_reward += self.reward
                self.done = True
                running = False
                print("Total Reward: ", self.total_reward)
                self.initialStates()
            
             # training
            self.agent.replay(batch_size,done)
            
            # update state
            GameState = NextState
            
            # epsilon greedy
            self.agent.adaptiveEGreedy()                       
                
            #draw / render(show)
            screen.fill(BLACK)
            self.player_group.draw(screen)
            self.enemy_group.draw(screen)

            #after drawing flip display
            pygame.display.flip()
                    
        pygame.quit()


# # Agent

# In[6]:


import random
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from collections import deque


# In[7]:


class DQLAgent:
    def __init__(self):
        # parameter / hyperparameter
        self.action_size = 4
        
        self.gamma = 0.95
        self.learning_rate = 0.001 
        
        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 3000) 
        
        self.model = self.build_model()
        
        
    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=4, strides = (2,2), input_shape = (IMGHEIGHT,IMGWIDTH,IMGHISTORY),padding = "same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64,kernel_size=4,strides=(2,2),padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64,kernel_size=3,strides=(1,1),padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dense(units= NBACTIONS, activation="linear"))
        
        model.compile(loss = "mse", optimizer="adam")
        
        return model
    
    def remember(self, state, action, reward, next_state): # (CaptureSample)
        # storage
        self.memory.append((state, action, reward, next_state))
    
    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size,done):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory,batch_size)
        
        inputs = np.zeros((batch_size,IMGHEIGHT,IMGWIDTH,IMGHISTORY))
        targets = np.zeros((inputs.shape[0],NBACTIONS))
        Q_sa = 0
        
        for i in range(len(minibatch)):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            next_state = minibatch[i][3]
            
            inputs[i:i + 1] = state
            targets[i]  = self.model.predict(state)
            Q_sa = self.model.predict(next_state)
            
            if done:
                targets[i,action] = reward
            else:
                targets[i,action] = reward + self.gamma*np.max(Q_sa)
                
            self.model.fit(inputs, targets ,batch_size= batch_size, epochs=1, verbose=0)
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# # Train Agent

# In[8]:


import numpy as np
import skimage as skimage
import warnings
import skimage.color
from skimage.transform import resize
warnings.filterwarnings("ignore")


# In[9]:


IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4


# In[10]:


def ProcessGameImage(RawImage):
    
    GreyImage = skimage.color.rgb2gray(RawImage)
    
    CroppedImage = GreyImage[0:HEIGHT,0:WIDTH]
    
    ReducedImage = skimage.transform.resize(CroppedImage,(IMGHEIGHT,IMGWIDTH))
    
    ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range = (0,255))
    
    ReducedImage = ReducedImage / 128
    
    return ReducedImage


# In[ ]:


if __name__ == '__main__':
    env = Env()
    liste = []
    t = 0
    testing = 1
    while testing:
        t += 1
        print("Episode:",t)
        liste.append(env.total_reward)
        
        # initialize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("kacma oyunu")
        clock = pygame.time.Clock()
        
        env.run()
        if t >= NUMEPISODES: testing = 0


# In[ ]:





# In[ ]:





# In[ ]:




