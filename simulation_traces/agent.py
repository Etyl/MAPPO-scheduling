import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
import os
import tqdm

from environment.scheduling_env import SchedulingEnv
from environment.constants import *

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(OBSERVATION_SPACE_SIZE, 128)), 
            nn.ReLU(),
            self._layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(128, 1))

        self.cov_var = torch.full(size=(num_actions,), fill_value=1.)
  
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).to(device)
        print("#")

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.float())
        logits = self.actor(hidden)
        mean = nn.Softmax(dim=-1)(logits) 
        dist = MultivariateNormal(mean, self.cov_mat)
        if action is None:
            action = dist.sample()
            action = torch.clamp(action, 0, 1)
            action = torch.nn.functional.normalize(action, p=1, dim=-1)

        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))

    use_saved_model = False

    save_file_learning = "./data/learning.csv"
    save_file_results = "./data/results.csv"
    with open(save_file_learning, "w") as f:
        f.write("episode-return, value-loss, policy-loss, old-approx-kl, approx-kl, clip-fraction, explained-variance\n")

    """ALGO PARAMS"""
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 1.0
    batch_size = 10
    max_cycles = 100
    total_episodes = 500

    """ ENV SETUP """
    env = SchedulingEnv()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).shape[0]
    observation_size = env.observation_space(env.possible_agents[0]).shape


    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
     # Load the saved agent
    if os.path.exists("./data/agent.pth") and use_saved_model:
        agent.load_state_dict(torch.load("./data/agent.pth"))
    
    optimizer = optim.Adam(agent.parameters(), lr=.001, eps=1e-5, betas=(0.999,0.999))
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = max_cycles
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, OBSERVATION_SPACE_SIZE)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents, num_actions)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in tqdm.tqdm(range(total_episodes)):

        agent.cov_mat = max((1-1.1*episode/(total_episodes-1)),0.001) * torch.diag(agent.cov_var).to(device)
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs, device)

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step-1)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        with open(save_file_learning, "a") as f:
            f.write(
                f"{np.mean(total_episodic_return)}, {v_loss.item()}, {pg_loss.item()}, {old_approx_kl.item()}, {approx_kl.item()}, {np.mean(clip_fracs)}, {explained_var.item()}\n"
            )

    """ SAVE THE MODEL """

    torch.save(agent.state_dict(), "./data/agent.pth")


    """ RENDER THE POLICY """
    env = SchedulingEnv()

    agent.eval()

    total_obs = []
    total_actions = []
    total_rewards = []
    total_infos = []
    total_energy = []

    with torch.no_grad():
        for episode in range(1):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            step = 0
            while not any(terms) and not any(truncs) and step < 1000:
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]

                step += 1
                total_obs.append(obs.cpu().numpy())
                total_actions.append(actions.cpu().numpy())
                total_rewards.append(list(rewards.values()))
                total_infos.append(np.array(infos["appLoad"]))
                total_energy.append(infos["energy"])

    with open(save_file_results, "w") as f:
        for (obs, actions, reward, info, energy) in zip(total_obs, total_actions,total_rewards,total_infos, total_energy):
            line =  ",".join(str(x) for x in obs.flatten()) + "," 
            line += ",".join(str(x) for x in actions.flatten()) + "," 
            line += ",".join(str(x) for x in info.flatten()) + ","
            line += str(energy) + ","
            line += ",".join(str(x) for x in reward) + "\n" 
            f.write(line)