import gym
import csv
import pdb
import numpy as np

from gym import utils
from copy import deepcopy
from .mujoco import MujocoEnv

""" Custom Hopper class for Gym Hopper """


class CustomHopper(MujocoEnv, utils.EzPickle):

    def __init__(self, domain=None, randomize=False, phi=None, dist=None, params=None):
        MujocoEnv.__init__(self, 4, randomize, phi, dist)
        utils.EzPickle.__init__(self)

        # default link masses
        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        self.phi = phi
        self.done = False
        self.debug = False
        self.params = params
        self.domain = domain
        self.randomize = randomize

        # torso mass shift from source to target environment
        if domain == 'source':  # (1kg shift)
            self.sim.model.body_mass[1] -= 1.0

        if params is not None:
            self.set_parameters(params)

    def set_debug(self, bool):
        self.debug = bool

    def set_randomness(self, dist):
        self.dist = dist

    def set_random_parameters(self):
        """ set random masses """
        self.set_parameters(*self.sample_parameters(phi=self.phi))
        if self.debug:
            self.print_parameters()

    def sample_parameters(self, phi):
        """ sample masses according to a domain distribution """
        if self.dist == "uniform":
            if self.params is not None:
                masses = [np.random.uniform(high=(1 - phi) * mass, low=(1 + phi) * mass)
                          for mass in self.params[1:]]
                masses.insert(0, self.sim.model.body_mass[1])
            else:
                masses = [np.random.uniform(high=(1 - phi) * mass, low=(1 + phi) * mass)
                          for mass in self.original_masses[1:]]
                masses.insert(0, self.sim.model.body_mass[1])
        if self.dist == "normal":
            if self.params is not None:
                masses = [np.random.normal(loc=mass, scale=phi)
                          for mass in self.params[1:]]
                masses.insert(0, self.sim.model.body_mass[1])
            else:
                masses = [np.random.normal(loc=mass, scale=phi)
                          for mass in self.original_masses[1:]]
                masses.insert(0, self.sim.model.body_mass[1])

        return masses

    def get_parameters(self):
        """ get mass value for each link """
        masses = np.array(self.sim.model.body_mass[1:])
        return masses

    def set_parameters(self, *task):
        params = np.clip(task, self.lower_bound, self.upper_bound)
        self.sim.model.body_mass[1:] = params

    def step(self, a):
        """ step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all()
                    and (np.abs(s[2:]) < 100).all()
                    and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        """ get current state """
        return np.concatenate([self.sim.data.qpos.flat[1:],
                               self.sim.data.qvel.flat])

    def reset_model(self):
        """ reset the environment to a random initial state """
        qpos = self.init_qpos + self.np_random.uniform(low=-.005,
                                                       high=.005,
                                                       size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005,
                                                       high=.005,
                                                       size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_mujoco_state(self, state):
        """ set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        """ set internal mujoco state """
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        """ returns current mjstate """
        return self.sim.get_state()

    def print_parameters(self):
        print(f'{self.domain}: {self.sim.model.body_mass}')

    def set_debug(self, on: bool = True):
        self.debug = on


""" Custom Hopper environments """

gym.envs.register(
    id="CustomHopper-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
    id="CustomHopper-source-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)

gym.envs.register(
    id="CustomHopper-source-UDR-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source", "randomize": True,
            "phi": 0.5, "dist": "uniform"}
)
