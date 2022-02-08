from dqn import weights_init, Initializer, DQN, HuberLoss, ReplayMemory
from pycar_env import PyCar
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
from collections import namedtuple

import math
import os
import random
import shutil
import warnings
import gym
from torch.backends import cudnn
from tqdm import tqdm

import logging
import time
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

class DQNAgent:

    def __init__(self):
        # self.config = config
        self.gamma = 0.75

        # self.logger = logging.getLogger("DQNAgent")

        self.screen_width = 600

        # define models (policy and target)
        self.policy_model = DQN()
        self.target_model = DQN()

        # define memory
        self.memory = ReplayMemory()

        # define loss
        self.loss = HuberLoss()

        # define optimizer
        self.optim = torch.optim.Adam(self.policy_model.parameters(), lr=0.0001)

        # define environment
        self.env = PyCar()#TODO
        # self.cartpole = PyCar(self.screen_width)

        # initialize counter
        self.current_episode = 0
        self.current_iteration = 0
        self.episode_durations = []

        self.batch_size = 250

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()

        self.cuda = self.is_cuda 

        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.policy_model = self.policy_model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Initialize Target model with policy model state dict
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        self.savepath = os.path.join(os.getcwd(), "model") + "/"
        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)

        t = time.localtime()
        self.save_tensorboard_path = os.path.join(os.getcwd(), "tensorboard_record") + "/run_" + time.strftime("%d_%m_%Y_%H_%M", t) + "/"
        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)
        self.writer = SummaryWriter(self.save_tensorboard_path)



    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt as e:
            print(e)

    def select_action(self, state, random_only=False):
        """
        The action selection function, it either uses the model to choose an action or samples one uniformly.
        :param state: current state of the model
        :return:
        """

        self.eps_start = 0.90
        self.eps_end = 0.35
        self.eps_decay = 500

        if self.cuda:
            state = state.cuda()
        sample = random.random()
        eps_threshold = self.eps_start - (self.eps_start - self.eps_end) * math.exp(
            -1. * self.current_iteration / self.eps_decay)

        self.writer.add_scalar('epsilon', eps_threshold, self.current_iteration)
        # print("Eps thresh: ", eps_threshold)
        if sample < eps_threshold and not random_only:
            # print("Model step")
            with torch.no_grad():
                return self.policy_model(state).max(1)[1].view(1, 1)  # size (1,1)
        else:
            # print("Random step")
            return torch.tensor([[random.randrange(5)]], device=self.device, dtype=torch.long)

    def get_action(self, state):

        if self.cuda:
            state = state.cuda()
        with torch.no_grad():
            return self.policy_model(state).max(1)[1].view(1, 1)  # size (1,1)
     
    def optimize_policy_model(self):
        """
        performs a single step of optimization for the policy model
        :return:
        """
        if self.memory.length() < self.batch_size:
            return

        self.memory.setup_epoch_training()

        total_loss = None
        training_len = math.ceil(self.memory.length()/self.batch_size)
        for i in range(training_len):
            # sample a batch
            transitions = self.memory.sample_batch(self.batch_size, i)
            len_transitions = len(transitions)

            one_batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, one_batch.next_state)), device=self.device,dtype=torch.uint8)  
            non_final_next_states = torch.cat([s for s in one_batch.next_state if s is not None]) 

            state_batch = torch.cat(one_batch.state)  
            action_batch = torch.cat(one_batch.action) 
            reward_batch = torch.cat(one_batch.reward)

            state_batch = state_batch.to(self.device)
            non_final_next_states = non_final_next_states.to(self.device)

            curr_state_values = self.policy_model(state_batch)  # [128, 2]
            curr_state_action_values = curr_state_values.gather(1, action_batch)  # [128, 1]

            next_state_values = torch.zeros(len_transitions, device=self.device)  # [128]
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()  # [< 128]

            # Get the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch  # [128]
            # compute loss: temporal difference error
            loss = self.loss(curr_state_action_values, expected_state_action_values.unsqueeze(1))

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

            # optimizer step
            self.optim.zero_grad()
            loss.backward()
            for param in self.policy_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optim.step()

        
        self.writer.add_scalar('loss', total_loss/training_len, self.current_iteration)        
        # return loss

    def train(self):
        """
        Training loop based on the number of episodes
        :return:
        """

        self.num_episodes = 2000
        self.target_update = 1

        mean_score, max_score, min_score = self.run_sim(100, random_only=True)

        self.writer.add_scalar('mean_score', mean_score, 0)
        self.writer.add_scalar('max_score', max_score, 0)
        self.writer.add_scalar('min_score', min_score, 0)

        for episode in tqdm(range(self.current_episode, self.num_episodes)):
            self.current_iteration += 1
            self.current_episode = episode
            # reset environment
            self.train_one_epoch()
            # The target network has its weights kept frozen most of the time
            if self.current_episode % self.target_update == 0:
                self.target_model.load_state_dict(self.policy_model.state_dict())

            if self.current_episode%25 == 0:
                torch.save(self.policy_model.state_dict(), self.savepath+"policy_epoch"+str(self.current_episode)+".pth")
                torch.save(self.target_model.state_dict(), self.savepath+"target_epoch"+str(self.current_episode)+".pth")

    def run_sim(self, count=20, random_only=False):
        score_list = []
        for i in range(count):
            self.env.reset_game()
            episode_duration = 0

            curr_state = torch.Tensor(self.env.get_state()).permute(2, 0, 1).unsqueeze(0)

            while(1):
                # time.sleep(0.1)
                episode_duration += 1

                # select action
                action = self.select_action(curr_state, random_only)

                images, reward, done,score = self.env.step(action.item())#TODO

                if self.cuda:
                    reward = torch.Tensor([reward]).to(self.device)
                else:
                    reward = torch.Tensor([reward]).to(self.device)

    
                # assign next state
                if done:
                    next_state = None
                else:
                    next_state = torch.Tensor(images).permute(2, 0, 1).unsqueeze(0) #TODO

                # add this transition into memory
                self.memory.push_transition(curr_state, action, next_state, reward)

                curr_state = next_state
                
                if done:
                    score_list.append(score)
                    break

        return np.mean(np.array(score_list)), np.max(np.array(score_list)), np.min(np.array(score_list)) 

    def train_one_epoch(self):
        """
        One episode of training; it samples an action, observe next screen and optimize the model once
        :return:
        """
        
        mean_score, max_score, min_score = self.run_sim()
        # print(mean_score)
        self.writer.add_scalar('mean_score', mean_score, self.current_iteration)
        self.writer.add_scalar('max_score', max_score, self.current_iteration)
        self.writer.add_scalar('min_score', min_score, self.current_iteration)

        # Policy model optimization step
        self.optimize_policy_model()
        

    def validate(self):
        
        curr_state = torch.Tensor(self.env.get_state()).permute(2, 0, 1).unsqueeze(0)

        while(1):
            # time.sleep(0.1)

            episode_duration += 1
            # select action
            action = self.get_action(curr_state)

            images, reward, done,score = self.env.step(action.item())#TODO

            if self.cuda:
                reward = torch.Tensor([reward]).to(self.device)
            else:
                reward = torch.Tensor([reward]).to(self.device)

 
            # assign next state
            if done:
                next_state = None
            else:
                next_state = torch.Tensor(images).permute(2, 0, 1).unsqueeze(0) #TODO

            curr_state = next_state
            
            if done:
                print(score)
                break

        # pass

if __name__=="__main__":

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    warnings.filterwarnings("ignore", category=UserWarning)

    agent = DQNAgent()
    agent.run()
    
