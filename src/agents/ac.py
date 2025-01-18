import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch.distributions import Normal


class Policy(torch.nn.Module):

    def __init__(self, state_space: int, action_space: int, actor=True,
                 seed: int = 42, hidden: int = 64, **kwargs):
        super().__init__()
        """ initializes a multi-layer neural network 
        to map observations s(t) from the environment into
        1. (actor): parameters of a normal distribution (mean μ(s(t)) and standard deviation σ(s(t))) 
           from which to sample the agent's actions -> π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))
        2. (critic): estimate of the state value function -> V(s(t))
               
        args:
            state_space: dimension of the observation space (environment)
            action_space: dimension of the action space (agent)
            actor: boolean condition to initialize 
                   -> the actor (actor=True)
                   -> the critic (actor=False)
        """
        torch.manual_seed(seed)

        self.action_space = action_space
        self.state_space = state_space
        self.tanh = torch.nn.Tanh()
        self.hidden = hidden
        self.actor = actor
        self.eps = 1e-6

        """ multi-layer neural network to output μ(s(t)) and V(s(t)) """
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, action_space)

        if actor:
            """ direct learnable parameter to output σ(s(t)) """
            init_sigma = 0.5
            self.sigma = torch.nn.Parameter(
                torch.zeros(self.action_space) + init_sigma)
            self.sigma_activation = F.softplus

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """ maps the observation x = s(t) from the environment at time-step t into 
        1. (actor): a normal distribution N(μ(s(t)), σ(s(t))) 
           from which to sample an agent action at time-step t
           -> π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))
        2. (critic): an estimate of the state value function at time-step t
           -> V(s(t))

        args:
            x: observation s(t) from the environment at time-step t 

        returns:
            (actor) normal_dist: normal distribution N(μ(s(t)) + ε, σ(s(t)) + ε)
                                 - μ(s(t)): action mean
                                 - σ(s(t)): action standard deviation
            (critic) action_mean: estimate of the state value function V(s(t))
        """
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        action_mean = self.fc3(x)

        if self.actor:
            sigma = self.sigma_activation(self.sigma)
            normal_dist = Normal(action_mean + self.eps, sigma + self.eps)
            return normal_dist
        else:
            return action_mean

    def to(self, device):
        """ move parameters to device """
        for param in self.parameters():
            param.data = param.data.to(device)
        return self


class A2CPolicy:

    def __init__(self, state_space: int, action_space: int,
                 seed: int = 42, **kwargs):
        """ initializes the multi-layer neural networks for the actor and the critic

        args:
            state_space: dimension of the observation space (environment)
            action_space: dimension of the action space (agent)
        """
        torch.manual_seed(seed)

        self.policies = OrderedDict()
        self.policies['actor'] = Policy(
            state_space, action_space, seed=seed, **kwargs)
        self.policies['critic'] = Policy(
            state_space, 1, actor=False, seed=seed, **kwargs)

    def to(self, device):
        """ move parameters to device """
        for k, v in self.policies.items():
            self.policies[k] = v.to(device)
        return self

    def state_dict(self):
        sd = OrderedDict()
        for k, nn in self.policies.items():
            sd[k] = nn.state_dict()
        return sd

    def load_state_dict(self, states, strict=True):
        for k, sd in states.items():
            self.policies[k].load_state_dict(sd, strict=strict)


class A2C:

    def __init__(self, policy,
                 device: str = 'cpu',
                 learning_rate: float = 2.5e-4,
                 max_grad_norm: float = 0.25,
                 entropy_coef: float = 0.25,
                 critic_coef: float = 0.5,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 seed: int = 42,
                 **kwargs):
        """ initializes an agent to learn a policy via Advantage Actor-Critic algorithm 

        args:
            policy: A2CPolicy
            device: processing device (e.g. CPU or GPU)
            learning_rate: learning rate for policy optimization
            max_grad_norm: maximum value for the gradient's L2 norm
            entropy_coef: entropy coefficient to balance exploration/exploitation
            critic_coef: critic coefficient to weight the value function loss
            batch_size: number of time-steps for policy update
            gamma: discount factor
        """
        torch.manual_seed(seed)

        self.device = device
        self.policy = policy.to(self.device)

        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.batch_size = batch_size
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(
            self.policy.policies['actor'].parameters(), lr=self.learning_rate)
        self.optimizer.add_param_group(
            {'params': self.policy.policies['critic'].parameters()})

        self.reset()

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        """ predicts an action based on observation s(t) and policy π(a(t)|s(t))
        -> a(t) ~ π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))

        args:
            obs: observation from the environment s(t)

        returns:
            action: action to perform -> a(t)
            action_log_prob: logarithmic probability value of the action -> log(π(a(t)|s(t)))
            state_value: estimate of the state value function V(s(t))
        """
        x = torch.from_numpy(obs).float().to(self.device)
        normal_dist = self.policy.policies['actor'](x)

        if deterministic:
            """ return the mean value of the policy π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))

            returns:
                a(t) = μ(s(t))
            """
            action = normal_dist.mean
            action = action.detach().cpu().numpy()
            return action, None

        else:
            """ sample an action from the policy π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))

            returns:
                a(t) ~ π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))
                log(π(a(t)|s(t)))
                V(s(t))
            """
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            state_value = self.policy.policies['critic'](x)
            action = action.detach().cpu().numpy()
            return action, action_log_prob, state_value

    def store_outcome(self, state, action_log_prob, reward, done, state_value):
        self.states.append(torch.from_numpy(state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.state_values.append(state_value)

    def update_policy(self):
        """ stack and move data to device """
        states = torch.stack(
            self.states, dim=0).to(self.device).squeeze(-1)
        action_log_probs = torch.stack(
            self.action_log_probs, dim=0).to(self.device).squeeze(-1)
        rewards = torch.tensor(
            self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(
            self.dones, dtype=torch.float32).to(self.device)
        state_values = torch.stack(
            self.state_values, dim=0).to(self.device).squeeze(-1)

        """ compute bootstrapped returns (backwards) """
        bootstrapped_returns = torch.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative * (1 - dones[t])
            bootstrapped_returns[t] = cumulative

        """ compute advantages via GAE """
        advantages = bootstrapped_returns - state_values
        advantages = (
            advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8)

        """ compute actor loss """
        actor_loss = - (action_log_probs * advantages.detach()).mean()

        """ compute critic loss """
        critic_loss = F.mse_loss(state_values, bootstrapped_returns.detach())

        """ compute total loss 
        with entropy regularization to encourage exploration """
        entropy = self.policy.policies['actor'](states).entropy().mean()
        loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy

        """ updates the policy network's weights """
        self.optimizer.zero_grad()
        loss.backward()

        """ clip gradients to prevent exploding gradients """
        torch.nn.utils.clip_grad_norm_(
            self.policy.policies['actor'].parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.reset()

    def reset(self):
        self.states = []
        self.action_log_probs = []
        self.rewards = []
        self.dones = []
        self.state_values = []
