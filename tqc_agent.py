import os
import time
import torch
import copy
import numpy as np
from tqc_model import Actor, Critic, quantile_huber_loss_f
from helper import write_into_file, mkdir, time_format
from replay_buffer import ReplayBuffer
# Building the whole Training Process into a class
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class TQC(object):
    def __init__(self, state_dim, action_dim, config):
        self.actor = Actor(state_dim, action_dim, config).to(config["device"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config["lr_actor"])
        self.critic = Critic(state_dim, action_dim, config).to(config["device"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config["lr_critic"])
        self.target_critic = Critic(state_dim, action_dim, config).to(config["device"])
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.device = config["device"]
        self.batch_size = int(config["batch_size"])
        self.discount = config["discount"]
        self.tau = config["tau"]
        self.device = config["device"]
        self.eval = 50
        self.write_tensorboard = False
        self.top_quantiles_to_drop = config["top_quantiles_to_drop_per_net"] * config["n_nets"]
        self.target_entropy = config["target_entropy"]
        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=config["device"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["lr_alpha"])
        self.total_it = 0
        self.step = 0
        self.seed = config["seed"]
        self.episodes = 100000
        self.locexp = str(config["locexp"])
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        pathname = dt_string + "seed_" + str(config["seed"])
        self.memory = ReplayBuffer((state_dim,), (action_dim,), int(config["buffer_size"]), self.seed, config["device"])
        tensorboard_name = str(config["locexp"]) + "/runs/" + pathname
        self.vid_path = str(config["locexp"]) + "/vid"
        self.writer = SummaryWriter(tensorboard_name)
        self.steps = 0
        self.start_timesteps = config["start_timesteps"]
        self.time_run = now.strftime("%d_%m_%Y_%H:%M:%S")

    def update(self, iterations):
        self.step += 1
        if self.step % 1000 == 0:
            self.write_tensorboard = 1 - self.write_tensorboard
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memoy
            # sys.stdout = open(os.devnull, "w")
            state, action, reward, next_state, done = self.memory.sample(self.batch_size)

            # sys.stdout = sys.__stdout__
            # for augment 1

            alpha = torch.exp(self.log_alpha)
            with torch.no_grad():
                # Step 5: Get policy action
                new_next_action, next_log_pi = self.actor(next_state)
                # compute quantile at next state
                next_z = self.target_critic(next_state, new_next_action)
                sorted_z, _ = torch.sort(next_z.reshape(self.batch_size, -1))
                sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
                target = reward + done * self.discount * (sorted_z_part - alpha * next_log_pi)
            # ---update critic
            cur_z = self.critic(state, action)
            critic_loss = quantile_huber_loss_f(cur_z, target, self.device)
            self.writer.add_scalar('Critic_loss', critic_loss, self.step)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # ---Update policy and alpha
            new_action, log_pi = self.actor(state)  # detached
            alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
            actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.writer.add_scalar('Actor_loss', actor_loss, self.step)
            self.actor_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.writer.add_scalar('Alpha_loss', actor_loss, self.step)
            self.alpha_optimizer.step()
            self.total_it += 1

    def select_action(self, obs):
        state = torch.as_tensor(obs, device=self.device, dtype=torch.float)
        # print(state.shape)
        if state.shape[0] != self.batch_size:
            state = state.unsqueeze(0)
        # print(state.shape)
        return self.actor.select_action(state)

    def save(self, filename):
        mkdir("", filename)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.log_alpha, filename + "_alpha")
        torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.log_alpha = torch.load(filename + "_alpha")
        self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))
        print("Loaded wights from ", filename)

    def train_agent(self, env):
        self.env = env
        scores_window = deque(maxlen=100)
        step_window = deque(maxlen=100)
        t = 0
        t0 = time.time()
        self.eval_policy(0)
        for i_epiosde in range(0, self.episodes):
            self.steps += 1
            episode_reward = 0
            state = self.env.reset()
            epi_step = 0
            for i in range(200):
                t += 1
                if t < self.start_timesteps:
                    action = self.env.action_space.sample()
                else:  # After 10000 timesteps, we switch to the model
                    # print(state.dtype)
                    action = self.select_action(state) * self.env.env_options.max_angle
                # action = 20
                # print("action", action)
                # print("state", state)
                next_state, reward, done, info = self.env.step(action)
                # print("state", next_state)
                episode_reward += reward
                if i_epiosde > 10:
                    self.update(1)
                self.memory.add(state, action, reward, next_state, done, done)
                state = next_state
                if done:
                    break
            if i_epiosde % self.eval == 0:
                self.save(self.locexp + "/models/model-{}".format(i_epiosde))
                self.eval_policy(i_epiosde)

            step_window.append(epi_step)
            scores_window.append(episode_reward)
            ave_reward = np.mean(scores_window)
            mean_steps = np.mean(np.array(step_window))
            print(
                "Epi {} Steps {} Re {:.2f} mean re {:.2f} Time {}".format(
                    i_epiosde,
                    epi_step,
                    episode_reward,
                    np.mean(scores_window),
                    time_format(time.time() - t0),
                )
            )
            self.writer.add_scalar("Aver_reward", ave_reward, self.steps)
            self.writer.add_scalar("steps_mean", mean_steps, self.steps)

    def eval_policy(self, eval_after_episode, episodes=1):
        """  """
        path = os.path.join(self.locexp, self.time_run)
        # path2 = "images-{}".format(eval_after_episode)
        # path = os.path.join(path1, path2)

        try:
            os.makedirs(path)
        except FileExistsError:
            print("path {} already exist".format(path))

        for i_episode in range(1, episodes + 1):
            episode_reward = 0
            state = self.env.reset()
            epi_step = 0
            while True:
                epi_step += 1
                action = self.select_action(state) * self.env.env_options.max_angle
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                if done:
                    break

            self.writer.add_scalar("eval_reward", episode_reward, self.steps)
            print("Eval reward {} eps step {}".format(episode_reward, epi_step))

    def eval_agent(self, env, path, episodes=1):
        print("Eval agent")
        self.load(path)
        for i_episode in range(1, episodes + 1):
            episode_reward = 0
            state = env.reset()
            epi_step = 0
            while True:
                epi_step += 1
                action = self.select_action(state) * env.env_options.max_angle
                state, reward, done, info = env.step(action)

                episode_reward += reward
                if done:
                    break
        #env.render()
        env.render_animated()
        self.writer.add_scalar("eval_reward", episode_reward, self.steps)
        print("Eval reward {} eps step {}".format(episode_reward, epi_step))
