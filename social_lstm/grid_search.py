import tensorflow as tf
from social_train import train
import argparse
import numpy as np


class GridSearch:

    def __init__(self, args=None, default=True):

        if args is None:
            args = []
        self.paramList = ["LR", "L2", "DR", "GC", "BS"]
        self.valueList = [[3E-3, 5E-4, 0.95, 10, 16]]
        if default:
            self.valueList.extend([
             [3E-3, 2E-4, 0.95, 10, 16], [3E-3, 1E-4, 0.95, 10, 16], [3E-3, 1E-3, 0.95, 10, 16], [3E-3, 2E-3, 0.95, 10, 16],
             [3E-3, 5E-4, 0.85, 10, 16], [3E-3, 5E-4, 0.90, 10, 16], [3E-3, 5E-4, 0.97, 10, 16], [3E-3, 5E-4, 0.99, 10, 16],
             [3E-3, 5E-4, 0.95, 5, 16], [3E-3, 5E-4, 0.95, 7, 16], [3E-3, 5E-4, 0.95, 20, 16], [3E-3, 5E-4, 0.95, 100, 16]])

        self.len = len(self.paramList)

        inputs = TrainInput(args.num_epochs, args.maxNumPeds)

        for values in self.valueList:
            # Initialize parameters
            hstring = self.get_hstring(values)
            inputs.set_learning_rate(values[0])
            inputs.set_lambda_param(values[1])
            inputs.set_decay_rate(values[2])
            inputs.set_grad_clip(values[3])
            inputs.set_batch_size(values[4])
            inputs.set_writer(hstring)
            # TODO inputs.set_leaveDataset
            print "Input parameters"
            for index in range(self.len):
                print self.paramList[index]+'='+str(values[index])

            # Do the training
            train(inputs)

            # Reset graph
            tf.reset_default_graph()

    def get_hstring(self, values):
        hstring = "lr="+str(values[0])
        for i in range(self.len-1):
            hstring = hstring+','+self.paramList[i+1]+'='+str(values[i+1])
        return hstring


class TrainInput:
    def __init__(self, num_epochs=1, maxNumPeds=40):
        self.rnn_size = 128
        self.num_layers = 1
        self.model = 'lstm'
        self.batch_size = 16
        self.seq_length = 12
        # self.num_epochs = 50
        self.num_epochs = num_epochs
        self.save_every = 400
        self.grad_clip = 10.
        self.learning_rate = 0.005
        self.decay_rate = 0.95
        self.keep_prob = 0.8
        self.embedding_size = 64
        self.neighborhood_size = 32
        self.grid_size = 4
        self.maxNumPeds = maxNumPeds
        self.leaveDataset = 4
        self.lambda_param = 0.0005
        self.writer = "training"
        self.dist_map = self.get_distMap()

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_lambda_param(self, l2):
        self.lambda_param = l2

    def set_decay_rate(self, dr):
        self.decay_rate = dr

    def set_grad_clip(self, gc):
        self.grad_clip = gc

    def set_batch_size(self, bs):
        self.batch_size = bs

    def set_writer(self, hstring):
        self.writer = hstring

    def set_leaveDataset(self, ds):
        self.leaveDataset = ds

    def get_distMap(self):
        # Neighborhood size
        ns = self.neighborhood_size
        # Half Neighborhood size
        hns = ns/2
        distMap = np.zeros([ns,ns],dtype=np.float)
        for x in range(ns):
            xnorm = (x-hns)**2
            for y in range(ns):
                ynorm = (y-hns)**2
                totalnorm = (xnorm+ynorm)
                if totalnorm > 1e-5:
                    distMap[x, y] = 1./totalnorm
        return distMap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # # RNN size parameter (dimension of the output/hidden state)
    # parser.add_argument('--rnn_size', type=int, default=128,
    #                     help='size of RNN hidden state')
    # # Number of layers parameter
    # parser.add_argument('--num_layers', type=int, default=1,
    #                     help='number of layers in the RNN')
    # # Model currently not used. Only LSTM implemented
    # # Type of recurrent unit parameter
    # parser.add_argument('--model', type=str, default='lstm',
    #                     help='rnn, gru, or lstm')
    # # Size of each batch parameter
    # parser.add_argument('--batch_size', type=int, default=16,
    #                     help='minibatch size')
    # # Length of sequence to be considered parameter
    # parser.add_argument('--seq_length', type=int, default=12,
    #                     help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of epochs')
    # # Frequency at which the model should be saved parameter
    # parser.add_argument('--save_every', type=int, default=400,
    #                     help='save frequency')
    # # Gradient value at which it should be clipped
    # parser.add_argument('--grad_clip', type=float, default=10.,
    #                     help='clip gradients at this value')
    # # Learning rate parameter
    # parser.add_argument('--learning_rate', type=float, default=0.005,
    #                     help='learning rate')
    # # Decay rate for the learning rate parameter
    # parser.add_argument('--decay_rate', type=float, default=0.95,
    #                     help='decay rate for rmsprop')
    # # Dropout not implemented.
    # # Dropout probability parameter
    # parser.add_argument('--keep_prob', type=float, default=0.8,
    #                     help='dropout keep probability')
    # # Dimension of the embeddings parameter
    # parser.add_argument('--embedding_size', type=int, default=64,
    #                     help='Embedding dimension for the spatial coordinates')
    # # Size of neighborhood to be considered parameter
    # parser.add_argument('--neighborhood_size', type=int, default=32,
    #                     help='Neighborhood size to be considered for social grid')
    # # Size of the social grid parameter
    # parser.add_argument('--grid_size', type=int, default=4,
    #                     help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=55,
                        help='Maximum Number of Pedestrians')
    # # The leave out dataset
    # parser.add_argument('--leaveDataset', type=int, default=3,
    #                     help='The dataset index to be left out in training')
    # # Lambda regularization parameter (L2)
    # parser.add_argument('--lambda_param', type=float, default=0.0005,
    #                     help='L2 regularization parameter')
    # # TensorBoard Writer name
    # parser.add_argument('--writer', type=str, default='training',
    #                     help='L2 regularization parameter')
    # # Obstacle Map
    # parser.add_argument('--obs_maps', type=list, default=[],
    #                     help='Obstacle Map file')
    args = parser.parse_args()
    GSearch = GridSearch(args)
