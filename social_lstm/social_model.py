'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 17th October 2016
'''

import tensorflow as tf
import numpy as np
# import pickle
from tensorflow.python.ops import rnn_cell
from grid import getSequenceGridMask
# import pdb


class SocialModel():
    def __init__(self, args, infer=False):
        '''
        Initialisation function for the class SocialModel
        params:
        args : Contains arguments required for the model creation
        '''

        # If sampling new trajectories, then infer mode
        if infer:
            # Sample one position at a time
            args.batch_size = 1
            args.seq_length = 1

        # Store the arguments
        self.args = args
        self.infer = infer

        # Store rnn size and grid_size
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size

        # Maximum number of peds
        self.maxNumPeds = args.maxNumPeds

        # NOTE : For now assuming, batch_size is always 1. That is the input
        # to the model is always a sequence of frames

        # Construct the basicLSTMCell recurrent unit with a dimension given by args.rnn_size
        with tf.name_scope("LSTM_cell"):
            cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)
            # if not infer and args.keep_prob < 1:
            # cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=args.keep_prob)

        # placeholders for the input data and the target data
        # A sequence contains an ordered set of consecutive frames
        # Each frame can contain a maximum of 'args.maxNumPeds' number of peds
        # For each ped we have their (pedID, x, y) positions as input
        self.input_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, 3], name="input_data")

        # target data would be the same format as input_data except with
        # one time-step ahead
        self.target_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, 3], name="target_data")

        # index of the obstacle map
        self.map_index = tf.placeholder(tf.int32, [1], name="map_index")

        # Grid data would be a binary matrix which encodes whether a pedestrian is present in
        # a grid cell of other pedestrian
        self.grid_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, args.maxNumPeds,
                                                     args.grid_size * args.grid_size], name="grid_data")

        # Variable to hold the value of the learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # Output dimension of the model
        self.output_size = 5
        self.mixture_size = args.num_dist

        # Define embedding and output layers
        self.define_embedding_and_output_layers(args)

        # Define LSTM states for each pedestrian
        with tf.variable_scope("LSTM_states"):
            self.LSTM_states = tf.zeros([args.maxNumPeds, cell.state_size], name="LSTM_states")
            self.initial_states = tf.split(self.LSTM_states, args.maxNumPeds, 0)

        # Define hidden output states for each pedestrian
        with tf.variable_scope("Hidden_states"):
            # self.output_states = tf.zeros([args.maxNumPeds, cell.output_size], name="hidden_states")
            self.output_states = tf.split(tf.zeros([args.maxNumPeds, cell.output_size]), args.maxNumPeds, 0)

        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
        with tf.name_scope("frame_data_tensors"):
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
            frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, args.seq_length, 0)]

        with tf.name_scope("frame_target_data_tensors"):
            # frame_target_data = tf.split(0, args.seq_length, self.target_data, name="frame_target_data")
            frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, args.seq_length, 0)]

        with tf.name_scope("grid_frame_data_tensors"):
            # This would contain a list of tensors each of shape MNP x MNP x (GS**2) encoding the mask
            # grid_frame_data = tf.split(0, args.seq_length, self.grid_data, name="grid_frame_data")
            grid_frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.grid_data, args.seq_length, 0)]

        # Cost
        with tf.name_scope("Cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")

        # Containers to store output distribution parameters
        with tf.name_scope("Distribution_parameters_stuff"):
            # self.initial_output = tf.zeros([args.maxNumPeds, self.output_size], name="distribution_parameters")
            self.initial_output = tf.split(tf.zeros([args.maxNumPeds, self.output_size*self.mixture_size]), args.maxNumPeds, 0)

        # Tensor to represent non-existent ped
        with tf.name_scope("Non_existent_ped_stuff"):
            nonexistent_ped = tf.constant(0.0, name="zero_ped")

        # Iterate over each frame in the sequence
        for seq, frame in enumerate(frame_data):
            print "Frame number", seq

            current_frame_data = frame  # MNP x 3 tensor
            current_grid_frame_data = grid_frame_data[seq]  # MNP x MNP x (GS**2) tensor
            # MNP x (GS**2 * RNN_size)
            social_tensor = self.getSocialTensor(current_frame_data, current_grid_frame_data, self.output_states)
            # NOTE: Using a tensor of zeros as the social tensor
            # social_tensor = tf.zeros([args.maxNumPeds, args.grid_size*args.grid_size*args.rnn_size])

            for ped in range(args.maxNumPeds):
                print "Pedestrian Number", ped

                # pedID of the current pedestrian
                pedID = current_frame_data[ped, 0]

                with tf.name_scope("extract_input_ped"):
                    # Extract x and y positions of the current ped
                    self.spatial_input = tf.slice(current_frame_data, [ped, 1], [1, 2])  # Tensor of shape (1,2)
                    # Extract the social tensor of the current ped
                    self.tensor_input = tf.slice(social_tensor, [ped, 0], [1,
                                                                           args.grid_size * args.grid_size * args.rnn_size])  # Tensor of shape (1, g*g*r)

                with tf.name_scope("embeddings_operations"):
                    # Embed the spatial input
                    embedded_spatial_input = tf.nn.relu(
                        tf.nn.xw_plus_b(self.spatial_input, self.embedding_w, self.embedding_b))
                    # Embed the tensor input
                    embedded_tensor_input = tf.nn.relu(
                        tf.nn.xw_plus_b(self.tensor_input, self.embedding_t_w, self.embedding_t_b))

                with tf.name_scope("concatenate_embeddings"):
                    # Concatenate the embeddings
                    complete_input = tf.concat([embedded_spatial_input, embedded_tensor_input], 1)

                # One step of LSTM
                with tf.variable_scope("LSTM") as scope:
                    if seq > 0 or ped > 0:
                        scope.reuse_variables()
                    self.output_states[ped], self.initial_states[ped] = cell(complete_input, self.initial_states[ped])

                # with tf.name_scope("reshape_output"):
                # Store the output hidden state for the current pedestrian
                #    self.output_states[ped] = tf.reshape(tf.concat(1, output), [-1, args.rnn_size])
                #    print self.output_states[ped]

                # Apply the linear layer. Output would be a tensor of shape 1 x output_size
                with tf.name_scope("output_linear_layer"):
                    self.initial_output[ped] = tf.nn.xw_plus_b(self.output_states[ped], self.output_w, self.output_b)

                # with tf.name_scope("store_distribution_parameters"):
                #    # Store the distribution parameters for the current ped
                #    self.initial_output[ped] = output

                with tf.name_scope("extract_target_ped"):
                    # Extract x and y coordinates of the target data
                    # x_data and y_data would be tensors of shape 1 x 1
                    [x_data, y_data] = tf.split(tf.slice(frame_target_data[seq], [ped, 1], [1, 2]), 2, 1)
                    target_pedID = frame_target_data[seq][ped, 0]

                with tf.name_scope("get_coef"):
                    # Extract coef from output of the linear output layer
                    [o_mu, o_s, o_alpha] = self.get_coef(self.initial_output[ped])
                    # print 'TEST!'
                    # print o_mux.get_shape()
                    # tf.summary.scalar("o_mux", tf.squeeze(o_mux))
                    # tf.summary.scalar("o_muy", tf.squeeze(o_muy))
                    # tf.summary.scalar("o_sx", tf.squeeze(o_sx))
                    # tf.summary.scalar("o_sy", tf.squeeze(o_sy))
                    # tf.summary.scalar("o_corr", tf.squeeze(o_corr))

                with tf.name_scope("calculate_loss"):
                    # Calculate loss for the current ped
                    lossfunc = self.get_lossfunc(o_mu, o_s, o_alpha, x_data, y_data)
                    # tf.summary.scalar("loss", lossfunc)

                with tf.name_scope("increment_cost"):
                    # If it is a non-existent ped, it should not contribute to cost
                    # If the ped doesn't exist in the next frame, he/she should not contribute to cost as well
                    self.cost = tf.where(
                        tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)),
                        self.cost, tf.add(self.cost, lossfunc))
                    self.counter = tf.where(
                        tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)),
                        self.counter, tf.add(self.counter, self.increment))
                    # tf.summary.scalar("cost", self.cost)
        with tf.name_scope("mean_cost"):
            # Mean of the cost
            self.cost = tf.div(self.cost, self.counter)

        # Get all trainable variables
        tvars = tf.trainable_variables()

        # L2 loss
        l2 = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars)
        self.cost = self.cost + l2
        # tf.summary.scalar("Cost", self.cost)
        # Get the final LSTM states
        self.final_states = tf.concat(self.initial_states, 0)

        # Get the final distribution parameters
        self.final_output = self.initial_output

        # Compute gradients
        self.gradients = tf.gradients(self.cost, tvars)

        # Clip the gradients
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # Define the optimizer
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        # The train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Merge all summmaries
        # self.summ = tf.summary.merge_all()

    def define_embedding_and_output_layers(self, args):
        # Define variables for the spatial coordinates embedding layer
        with tf.variable_scope("coordinate_embedding"):
            self.embedding_w = tf.get_variable("embedding_w", [2, args.embedding_size],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_b = tf.get_variable("embedding_b", [args.embedding_size],
                                               initializer=tf.constant_initializer(0.1))
            # tf.summary.histogram("weights", self.embedding_w)
            # tf.summary.histogram("biases", self.embedding_b)

        # Define variables for the social tensor embedding layer
        with tf.variable_scope("tensor_embedding"):
            self.embedding_t_w = tf.get_variable("embedding_t_w",
                                                 [args.grid_size * args.grid_size * args.rnn_size, args.embedding_size],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_t_b = tf.get_variable("embedding_t_b", [args.embedding_size],
                                                 initializer=tf.constant_initializer(0.1))
            # tf.summary.histogram("weights", self.embedding_t_w)
            # tf.summary.histogram("biases", self.embedding_t_b)

        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", [args.rnn_size, self.output_size*self.mixture_size],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.output_b = tf.get_variable("output_b", [self.output_size*self.mixture_size], initializer=tf.constant_initializer(0.1))
            # tf.summary.histogram("weights", self.output_w)
            # tf.summary.histogram("biases", self.output_b)

    def tf_2d_normal(self, x, y, mux, muy, sx, sy, rho):
        '''
        Function that implements the PDF of a 2D normal distribution
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # eq 3 in the paper
        # and eq 24 & 25 in Graves (2013)
        # Calculate (x - mux) and (y-muy)
        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)
        # Calculate sx*sy
        sxsy = tf.multiply(sx, sy)
        # Calculate the exponential factor
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2 * tf.div(
            tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
        negRho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.div(-z, 2 * negRho))
        # Normalization constant
        denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    def get_lossfunc(self, z_mu, z_s, z_alpha, x_data, y_data):
        '''
        Function to calculate given a 2D distribution over x and y, and target data
        of observed x and y points
        params:
        z_mu : mean of the distribution in x and y
        z_s : std dev of the distribution in x and y
        z_rho : Correlation factor of the distribution
        x_data : target x points
        y_data : target y points
        '''
        data = tf.squeeze(tf.stack([x_data, y_data], axis=1), [2])

        # Calculate the Log PDF of the data w.r.t to the distribution
        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=z_mu, scale_diag=z_s)
        pdf = dist.prob(data)
        pdf = tf.expand_dims(pdf, 1)

        # weighting the biv. gaussian distributions
        result = tf.matmul(z_alpha, pdf)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(result)

        return result

    def get_coef(self, output):
        # eq 20 -> 22 of Graves (2013)

        z = output
        # Split the output into 5 parts corresponding to means, std devs and corr
        z_mux, z_muy, z_sx, z_sy, z_alph = tf.split(z, self.output_size, 1)
        # The output must be exponentiated for the std devs
        z_sx = tf.exp(z_sx)
        z_sy = tf.exp(z_sy)
        z_mu = tf.stack([tf.squeeze(z_mux), tf.squeeze(z_muy)], axis=1)
        z_s = tf.stack([tf.squeeze(z_sx), tf.squeeze(z_sy)], axis=1)
        # Tanh applied to keep it in the range [-1, 1]
        # z_alph = tf.nn.softmax(z_alph)

        max_alph = tf.reduce_max(z_alph, 1, keep_dims=True)
        alph = tf.subtract(z_alph, max_alph)

        z_alph = tf.exp(z_alph)

        normalize_alph = tf.reciprocal(tf.reduce_sum(z_alph, 1, keep_dims=True))
        z_alph = tf.mul(normalize_alph, z_alph)

        return [z_mu, z_s, z_alph]

    def get_coef0(self, output):
        # eq 20 -> 22 of Graves (2013)

        z = output
        # Split the output into 5 parts corresponding to means, std devs and corr
        z_mux, z_muy, z_sx, z_sy, z_alph = tf.split(z, self.output_size, 0)
        # The output must be exponentiated for the std devs
        z_sx_0 = tf.exp(z_sx[0])
        z_sy_0 = tf.exp(z_sy[0])

        z_mu = tf.stack([tf.squeeze(z_mux[0]), tf.squeeze(z_muy[0])], axis=1)
        z_s = tf.stack([tf.squeeze(z_sx_0), tf.squeeze(z_sy_0)], axis=1)

        # Tanh applied to keep it in the range [-1, 1]
        z_alp = tf.nn.softmax(z_alph)

        return [z_mu, z_s, z_alp]

    def getSocialTensor(self, current_frame_data, grid_frame_data, output_states):
        '''
        Computes the social tensor for all the maxNumPeds in the frame
        params:
        grid_frame_data : A tensor of shape MNP x MNP x (GS**2)
        output_states : A list of tensors each of shape 1 x RNN_size of length MNP
        '''
        # Create a zero tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.zeros([self.args.maxNumPeds, self.grid_size * self.grid_size, self.rnn_size],
                                 name="social_tensor")
        # Create a list of zero tensors each of shape 1 x (GS**2) x RNN_size of length MNP
        social_tensor = tf.split(social_tensor, self.args.maxNumPeds, 0)
        # Concatenate list of hidden states to form a tensor of shape MNP x RNN_size
        hidden_states = tf.concat(output_states, 0)
        # Split the grid_frame_data into grid_data for each pedestrians
        # Consists of a list of tensors each of shape 1 x MNP x (GS**2) of length MNP
        grid_frame_ped_data = tf.split(grid_frame_data, self.args.maxNumPeds, 0)
        # Squeeze tensors to form MNP x (GS**2) matrices
        grid_frame_ped_data = [tf.squeeze(input_, [0]) for input_ in grid_frame_ped_data]

        # For each pedestrian
        for ped in range(self.args.maxNumPeds):
            # Compute social tensor for the current pedestrian
            with tf.name_scope("tensor_calculation"):

                social_tensor_ped = tf.matmul(tf.transpose(grid_frame_ped_data[ped]), hidden_states)
                social_tensor[ped] = tf.reshape(social_tensor_ped, [1, self.grid_size * self.grid_size, self.rnn_size])

        # Concatenate the social tensor from a list to a tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.concat(social_tensor, 0)
        # Reshape the tensor to match the dimensions MNP x (GS**2 * RNN_size)
        social_tensor = tf.reshape(social_tensor,
                                   [self.args.maxNumPeds, self.grid_size * self.grid_size * self.rnn_size])
        return social_tensor

    def sample_gaussian_2d(self, mux, muy, sx, sy, rho):
        '''
        Function to sample a point from a given 2D normal distribution
        params:
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # Extract mean
        mean = [mux, muy]
        # Extract covariance matrix
        cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
        # Sample a point from the multivariate normal distribution
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    def sample_biv_gaussian(self, mu, s, alph):
        '''
        Function to sample a point from a given Diagonal Bivariate normal distribution
        params:
        mu : mean of the distribution in x and y
        s : std dev of the distribution in x and y
        alph : Weighting factors of the mixture
        '''

        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=s)
        # Sampling the distributions
        samples = dist.sample()
        samples = tf.expand_dims(samples, 1)
        # weighting the biv. gaussian distributions
        result = tf.squeeze(tf.matmul(alph, samples))

        return result

    def sample(self, sess, traj, grid, dimensions, true_traj, num=10):
        # traj is a sequence of frames (of length obs_length)
        # so traj shape is (obs_length x maxNumPeds x 3)
        # grid is a tensor of shape obs_length x maxNumPeds x maxNumPeds x (gs**2)
        states = sess.run(self.LSTM_states)
        # writer = tf.summary.FileWriter('save/sample')
        # writer.add_graph(sess.graph)
        # print "Fitting"
        # For each frame in the sequence
        for index, frame in enumerate(traj[:-1]):
            data = np.reshape(frame, (1, self.maxNumPeds, 3))
            target_data = np.reshape(traj[index + 1], (1, self.maxNumPeds, 3))
            grid_data = np.reshape(grid[index, :],
                                   (1, self.maxNumPeds, self.maxNumPeds, self.grid_size * self.grid_size))

            feed = {self.input_data: data, self.LSTM_states: states, self.grid_data: grid_data,
                    self.target_data: target_data}

            [states, cost] = sess.run([self.final_states, self.cost], feed)
            # writer.add_summary(s, index)
            # print cost

        ret = traj

        last_frame = traj[-1]

        prev_data = np.reshape(last_frame, (1, self.maxNumPeds, 3))
        prev_grid_data = np.reshape(grid[-1], (1, self.maxNumPeds, self.maxNumPeds, self.grid_size * self.grid_size))

        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, self.maxNumPeds, 3))
        # Prediction
        for t in range(num):
            # print "**** NEW PREDICTION TIME STEP", t, "****"
            feed = {self.input_data: prev_data, self.LSTM_states: states, self.grid_data: prev_grid_data,
                    self.target_data: prev_target_data}
            [output, states, cost] = sess.run([self.final_output, self.final_states, self.cost], feed)
            # writer.add_summary(s, t)
            # print "Cost", cost
            # Output is a list of lists where the inner lists contain matrices of shape 1x5. The outer list contains only one element (since seq_length=1) and the inner list contains maxNumPeds elements
            # output = output[0]
            newpos = np.zeros((1, self.maxNumPeds, 3))
            for pedindex, pedoutput in enumerate(output):
                mu, s, alph = self.get_coef0(pedoutput[0])
                next_x, next_y = self.sample_biv_gaussian(mu, s, alph)

                # if prev_data[0, pedindex, 0] != 0:
                #     print "Pedestrian ID", prev_data[0, pedindex, 0]
                #     print "Predicted parameters", mux, muy, sx, sy, corr
                #     print "New Position", next_x, next_y
                #     print "Target Position", prev_target_data[0, pedindex, 1], prev_target_data[0, pedindex, 2]
                #     print

                newpos[0, pedindex, :] = [prev_data[0, pedindex, 0], next_x, next_y]
            ret = np.vstack((ret, newpos))
            print "ret"
            print ret.get_shape()
            prev_data = newpos
            prev_grid_data = getSequenceGridMask(prev_data, dimensions, self.args.neighborhood_size, self.grid_size)
            if t != num - 1:
                prev_target_data = np.reshape(true_traj[traj.shape[0] + t + 1], (1, self.maxNumPeds, 3))

        # The returned ret is of shape (obs_length+pred_length) x maxNumPeds x 3
        return ret

    def sample_nocost(self, sess, traj, grid, dimensions, num=10):
        # traj is a sequence of frames (of length obs_length)
        # so traj shape is (obs_length x maxNumPeds x 3)
        # grid is a tensor of shape obs_length x maxNumPeds x maxNumPeds x (gs**2)
        states = sess.run(self.LSTM_states)
        # writer = tf.summary.FileWriter('save/Ownsample')
        # writer.add_graph(sess.graph)
        # print "Fitting"
        # For each frame in the sequence
        for index, frame in enumerate(traj[:-1]):
            data = np.reshape(frame, (1, self.maxNumPeds, 3))
            target_data = np.reshape(traj[index + 1], (1, self.maxNumPeds, 3))
            grid_data = np.reshape(grid[index, :],
                                   (1, self.maxNumPeds, self.maxNumPeds, self.grid_size * self.grid_size))

            feed = {self.input_data: data, self.LSTM_states: states, self.grid_data: grid_data,
                    self.target_data: target_data}

            [states, cost, s] = sess.run([self.final_states, self.cost, self.summ], feed)
            # writer.add_summary(s, index)
            # print cost

        ret = traj

        last_frame = traj[-1]

        prev_data = np.reshape(last_frame, (1, self.maxNumPeds, 3))
        prev_grid_data = np.reshape(grid[-1], (1, self.maxNumPeds, self.maxNumPeds, self.grid_size * self.grid_size))

        # Prediction
        for t in range(num):
            # print "**** NEW PREDICTION TIME STEP", t, "****"
            feed = {self.input_data: prev_data, self.LSTM_states: states, self.grid_data: prev_grid_data}
            [output, states, s] = sess.run([self.final_output, self.final_states, self.summ], feed)
            # writer.add_summary(s, t)
            # print "Cost", cost
            # Output is a list of lists where the inner lists contain matrices of shape 1x5. The outer list contains only one element (since seq_length=1) and the inner list contains maxNumPeds elements
            # output = output[0]
            newpos = np.zeros((1, self.maxNumPeds, 3))
            for pedindex, pedoutput in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(pedoutput[0], 5, 0)
                mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])

                next_x, next_y = self.sample_gaussian_2d(mux, muy, sx, sy, corr)

                # if prev_data[0, pedindex, 0] != 0:
                #     print "Pedestrian ID", prev_data[0, pedindex, 0]
                #     print "Predicted parameters", mux, muy, sx, sy, corr
                #     print "New Position", next_x, next_y
                #     print

                newpos[0, pedindex, :] = [prev_data[0, pedindex, 0], next_x, next_y]
            ret = np.vstack((ret, newpos))
            print "ret_val"
            print ret.get_shape()
            prev_data = newpos
            prev_grid_data = getSequenceGridMask(prev_data, dimensions, self.args.neighborhood_size, self.grid_size)

        # The returned ret is of shape (obs_length+pred_length) x maxNumPeds x 3
        return ret
