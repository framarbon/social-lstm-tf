import tensorflow as tf
from social_train import train


class GridSearch:

    def __init__(self, default=True):

        self.paramList = ["LR", "L2", "DR", "GC", "BS"]
        self.valueList = [[3E-3, 5E-4, 0.95, 10, 16]]
        if default:
            self.valueList.extend([
             [3E-3, 2E-4, 0.95, 10, 16], [3E-3, 1E-4, 0.95, 10, 16], [3E-3, 1E-3, 0.95, 10, 16], [3E-3, 2E-3, 0.95, 10, 16],
             [3E-3, 5E-4, 0.85, 10, 16], [3E-3, 5E-4, 0.90, 10, 16], [3E-3, 5E-4, 0.97, 10, 16], [3E-3, 5E-4, 0.99, 10, 16],
             [3E-3, 5E-4, 0.95, 5, 16], [3E-3, 5E-4, 0.95, 7, 16], [3E-3, 5E-4, 0.95, 20, 16], [3E-3, 5E-4, 0.95, 100, 16]])

        self.len = len(self.paramList)

        inputs = TrainInput()

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


class TrainInput():
    def __init__(self):
        self.rnn_size = 128
        self.num_layers = 1
        self.model = 'lstm'
        self.batch_size = 16
        self.seq_length = 12
        # self.num_epochs = 50
        self.num_epochs = 1
        self.save_every = 400
        self.grad_clip = 10.
        self.learning_rate = 0.005
        self.decay_rate = 0.95
        self.keep_prob = 0.8
        self.embedding_size = 64
        self.neighborhood_size = 32
        self.grid_size = 4
        self.maxNumPeds = 40
        self.leaveDataset = 3
        self.lambda_param = 0.0005
        self.writer = "training"

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

if __name__ == '__main__':
    GSearch = GridSearch()
