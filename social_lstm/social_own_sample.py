'''
Interface to test the trained model on custom scenarios

Author : Francisco Marquez bonilla
Date : 6th September 2017
'''

import numpy as np
import tensorflow as tf

import os
import pickle

from social_model import SocialModel
from grid import getSequenceGridMask


def ownsample(obs_traj, pred_length=4, save_directory='save/1/'):

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Create a SocialModel object with the saved_args and infer set to true
    model = SocialModel(saved_args, True)
    # Initialize a TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize a saver
    saver = tf.train.Saver()

    # Get the checkpoint state for the model
    ckpt = tf.train.get_checkpoint_state('save')
    print ('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    dimensions = [720, 576]
    obs_grid = getSequenceGridMask(obs_traj, [720, 576], saved_args.neighborhood_size, saved_args.grid_size)

    complete_traj = model.sample_nocost(sess, obs_traj, obs_grid, dimensions, pred_length)

    print "Mean error of the model on this scenario is unknown",

    return complete_traj

