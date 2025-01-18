import torch
import torch.nn.functional as F

from torch.distributions import Normal


class RFPolicy(torch.nn.Module):

    def __init__(self, state_space: int, action_space: int,
                 seed: int = 42, hidden: int = 64, **kwargs):
        super().__init__()
        """ initializes a multi-layer neural network 
        to map observations s(t) from the environment into
        parameters of a normal distribution (mean μ(s(t)) and standard deviation σ(s(t))) 
        from which to sample the agent's actions -> π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))
        
        args:
            state_space: dimension of the observation space (environment)
            action_space: dimension of the action space (agent)
        """
        torch.manual_seed(seed)

        self.action_space = action_space
        self.state_space = state_space
        self.tanh = torch.nn.Tanh()
        self.hidden = hidden
        self.eps = 1e-6

        """ multi-layer neural network to output μ(s(t)) """
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, action_space)

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
        a normal distribution N(μ(s(t)), σ(s(t))) 
        from which to sample an agent action at time-step t
        -> π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))

        args:
            x: observation s(t) from the environment at time-step t 

        returns:
            (actor) normal_dist: normal distribution N(μ(s(t)) + ε, σ(s(t)) + ε)
                                 - μ(s(t)): action mean
                                 - σ(s(t)): action standard deviation
        """
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        action_mean = self.fc3(x)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean + self.eps, sigma + self.eps)
        return normal_dist

    def to(self, device):
        """ move parameters to device """
        for param in self.parameters():
            param.data = param.data.to(device)
        return self


class RF:

    def __init__(self, policy,
                 device: str = 'cpu',
                 baseline: str = 'vanilla',
                 learning_rate: float = 2.5e-4,
                 max_grad_norm: float = 0.25,
                 gamma: float = 0.99,
                 seed: int = 42,
                 **kwargs):
        """ initializes an agent to learn a policy via REINFORCE algorithm 

        args:
            policy: RFPolicy
            device: processing device (e.g. CPU or GPU)
            baseline: algorithm baseline
            learning_rate: learning rate for policy optimization
            max_grad_norm: maximum value for the gradient's L2 norm
            gamma: discount factor
        """
        torch.manual_seed(seed)

        self.device = device
        self.policy = policy.to(self.device)

        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.baseline = baseline
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.learning_rate)

        self.reset()

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        """ predicts an action based on observation s(t) and policy π(a(t)|s(t))
        -> a(t) ~ π(a(t)|s(t)) = N(μ(s(t)), σ(s(t)))

        args:
            obs: observation from the environment s(t)

        returns:
            action: action to perform -> a(t)
            action_log_prob: logarithmic probability value of the action -> log(π(a(t)|s(t)))
        """
        x = torch.from_numpy(obs).float().to(self.device)
        normal_dist = self.policy(x)

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
            """
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            action = action.detach().cpu().numpy()
            return action, action_log_prob

    def store_outcome(self, state, action_log_prob, reward):
        self.states.append(torch.from_numpy(state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        """ stack and move data to device """
        states = torch.stack(
            self.states, dim=0).to(self.device).squeeze(-1)
        action_log_probs = torch.stack(
            self.action_log_probs, dim=0).to(self.device).squeeze(-1)
        rewards = torch.tensor(
            self.rewards, dtype=torch.float32).to(self.device)

        """ compute discounted returns (backwards) """
        discounted_returns = torch.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            discounted_returns[t] = cumulative

        """ enforce baseline """
        if self.baseline == 'constant':
            discounted_returns -= 20
        if self.baseline == 'whitening':
            discounted_returns = (
                discounted_returns - discounted_returns.mean()) / (
                    discounted_returns.std() + 1e-8)

        """ compute actor loss """
        loss = - (action_log_probs * discounted_returns).mean()

        """ updates the policy network's weights """
        self.optimizer.zero_grad()
        loss.backward()

        """ clip gradients to prevent exploding gradients """
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.reset()

    def reset(self):
        self.states = []
        self.action_log_probs = []
        self.rewards = []
