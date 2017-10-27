'''
Interface to test the trained model on custom scenarios

Author : Francisco Marquez bonilla
Date : 6th September 2017
'''

import tensorflow as tf
import os
import pickle
from social_model import SocialModel
from grid import getSequenceGridMask


class MySampler:
    def __init__(self, save_directory='save/1/'):
    # Define the path for the config file for saved args
        with open(os.path.join(save_directory, 'social_config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)

        # TODO: initialize TF graph in python ros node
        tf.reset_default_graph()

        # Create a SocialModel object with the saved_args and infer set to true
        self.model = SocialModel(self.saved_args, True)
        # Initialize a TensorFlow session
        self.sess = tf.InteractiveSession()
        # Initialize a saver
        self.saver = tf.train.Saver()

        # Get the checkpoint state for the model
        ckpt = tf.train.get_checkpoint_state('save/1')


        # Restore the model at the checkpoint
        print ('loading model: ', ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        # print ('loading model: ', ckpt.all_model_checkpoint_paths[40])
        # saver.restore(sess, ckpt.all_model_checkpoint_paths[40])

    def mysample(self, obs_traj, pred_length=4,):

        dimensions = [720, 576]
        obs_grid = getSequenceGridMask(obs_traj, [720, 576], self.saved_args.neighborhood_size, self.saved_args.grid_size)

        print "********************** SAMPLING A NEW TRAJECTORY******************************"
        complete_traj = self.model.sample_nocost(self.sess, obs_traj, obs_grid, dimensions, pred_length)

        print "Mean error of the model on this scenario is unknown",

        return complete_traj

