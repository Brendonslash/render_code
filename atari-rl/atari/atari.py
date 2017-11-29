import cv2
import numpy as np
import time
import util

from gym.envs.atari.atari_env import AtariEnv
import os
import numpy as np
from gym.envs.atari import AtariEnv
from gym.core import Wrapper
from gym import utils, spaces, error
import atari_py
from ale_python_interface import ALEInterface
from time import gmtime, strftime
import threading
import time

import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import widgets
from IPython.display import display
import tensorflow as tf
import gym
from gym import wrappers
import random

from matplotlib import animation
#from JSAnimation.IPython_display import display_animation
def display_frames(frames, filename_gif = None):
    """
    Displays a list of frames as a gif, with controls
    """
    filename_gif = 'temp.gif'
    plt.figure(figsize=(frames[0].shape[1] / 18.0, frames[0].shape[0] / 18.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=10)
    if filename_gif: 
        anim.save(filename_gif, writer = 'imagemagick', fps=20)
    
    
class Atari(object):
  def __init__(self, summary, config):
    self.summary = summary

    util.log('Starting %s {frameskip: %s, repeat_action_probability: %s}' %
             (config.game, str(config.frameskip),
              str(config.repeat_action_probability)))

    self.env = Atari.create_env(config)
    self.env.mode = 'fast'
    if isinstance(config.frameskip, int):
      frameskip = config.frameskip
    else:
      frameskip = config.frameskip[1]

    self.input_frames = config.input_frames
    self.input_shape = config.input_shape
    self.max_noops = config.max_noops / frameskip
    self.render = config.render

    config.num_actions = self.env.action_space.n
    self.episode = 0

  def sample_action(self):
    return self.env.action_space.sample()

  def reset(self):
    """Reset the game and play some random actions"""

    self.start_time = time.time()
    self.episode += 1
    self.steps = 0
    self.score = 0

    self.last_frame = self.env.reset()
    if self.render: self.env.render()
    self.frames = []

    for i in range(np.random.randint(self.input_frames, self.max_noops + 1)):
      frame, reward_, done, _ = self.env.step(0)
      if self.render: self.env.render()

      self.steps += 1
      self.frames.append(self.process_frame(self.last_frame, frame))
      self.last_frame = frame
      self.score += reward_

      if done: self.reset()
      
    return self.frames[-self.input_frames:], self.score, done

  def step(self, action):
    frame, reward, done, _ = self.env.step(action)
    if self.render: self.env.render()

    self.steps += 1
    self.frames.append(self.process_frame(self.last_frame, frame))
    self.last_frame = frame
    self.score += reward
    
    return self.frames[-self.input_frames:], reward, done

  def process_frame(self, last_frame, current_frame):
    # Max last 2 frames to remove flicker
    # TODO Use ALE color_averaging instead (current causes core dump)
    frame = np.maximum(last_frame, current_frame)

    # Rescale image
    frame = cv2.resize(frame, self.input_shape)

    # Convert to greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame

  def log_episode(self, step):
    duration = time.time() - self.start_time
    steps_per_sec = self.steps / duration
    display_frames(self.frames)
    message = 'Episode %d, score %.0f (%d steps, %.2f secs, %.2f steps/sec)'
    util.log(message %
             (self.episode, self.score, self.steps, duration, steps_per_sec))

    self.summary.episode(step, self.score, self.steps, duration)

  @classmethod
  def create_env(cls, config):
    return AtariEnv(
        game=config.game,
        obs_type='image',
        frameskip=config.frameskip,
        repeat_action_probability=config.repeat_action_probability)

  @classmethod
  def num_actions(cls, config):
    return Atari.create_env(config).action_space.n


class FastAtariEnv(AtariEnv):
  def __init__(self, game='Breakout', obs_type='image', frameskip=(2, 5), repeat_action_probability=0.):
      self.game_path = atari_py.get_game_path(game)
      self._obs_type = obs_type
      self.frameskip = frameskip
      self.ale = ALEInterface()
      self.viewer = None
      assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
      self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)
      self._seed()
      (screen_width, screen_height) = self.ale.getScreenDims()
      self._buffer = np.empty((screen_height, screen_width, 3), dtype=np.uint8)
  def _get_image(self):
    # Don't reorder from rgb to bgr as we're converting to greyscale anyway
    self.ale.getScreenRGB(self._buffer)  # says rgb but actually bgr
    return self._buffer
