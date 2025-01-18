import os
import gym
import numpy as np
from os import path

from gym import error, spaces
from gym.utils import seeding
from collections import OrderedDict

try:
    import mujoco_py

except ImportError as e:
    raise error.DependencyNotInstalled(
        "requirement: install mujoco_py (https://mujoco.org/)")


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()]))

    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, - float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)

    else:
        raise NotImplementedError(type(observation), observation)

    return space


""" wrapper class for MuJoCo environments """


class MujocoEnv(gym.Env):
    """ interface for MuJoCo environments """

    def __init__(self, frame_skip, randomize: bool = False, phi: float = None, dist: str = None):

        self.frame_skip = frame_skip
        self.build_model()
        self.data = self.sim.data
        self.randomize = randomize

        if self.randomize:  # randomization parameters
            self.phi = phi
            self.dist = dist

        self.lower_bound = 0.01
        self.upper_bound = 10.0

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def build_model(self):
        self.model = mujoco_py.load_model_from_path(os.path.join(os.path.dirname(__file__),
                                                                 "assets/hopper.xml"))
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None
        self._viewers = {}

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """ reset the robot degrees of freedom (qpos and qvel) """
        raise NotImplementedError

    def viewer_setup(self):
        """ method to tinker with camera position (at viewer initialization) """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.randomize:
            self.set_random_parameters()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (
            self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=500,
               height=500,
               camera_id=None,
               camera_name=None):

        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Warning: do not specify `camera_id` and `camera_name` at the same time")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=False)
            # flip the original image (upside down)
            return data[::-1, :, :]

        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size for old mujoco-py:
            # extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=True)[1]
            # flip the original image (upside down)
            return data[::-1, :]

        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
