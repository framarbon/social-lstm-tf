import tensorflow as tf
import argparse
import os
import time
import pickle
import numpy as np

from social_model import SocialModel
from social_utils import SocialDataLoader
from grid import getSequenceGridMask


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=12,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Number of gaussians dist
    parser.add_argument('--num_dist', type=int, default=4,
                        help='number of distribution')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=10,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=12,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=40,
                        help='Maximum Number of Pedestrians')
    # The training dataset
    parser.add_argument('-t', '--trainingDataset', type=int, nargs='+', default=[0,1,2,3,4,5])
    # The validation dataset
    parser.add_argument('-v', '--validDataset', type=int, default=2)

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    # TensorBoard Writer name
    parser.add_argument('--writer', type=str, default='training',
                        help='L2 regularization parameter')
    # Save path
    parser.add_argument('--save_path', type=str, default="",
                        help='save_path')
    args = parser.parse_args()
    train(args)


def train(args):
    datasets = args.trainingDataset
    v_index = []
    if args.validDataset >= 0:
        v_index = datasets[-args.validDataset:]
    print "Dataset used for training: "+str(datasets[:-args.validDataset])
    print "Dataset used for validation: "+str(v_index)

    if not args.save_path:
        args.save_path = os.getcwd()

    # Create the SocialDataLoader object
    data_loader = SocialDataLoader(args.batch_size, args.seq_length, args.maxNumPeds, datasets, forcePreProcess=True, infer=False, valid_index=v_index)

    # Log directory
    log_directory = 'log/'
    log_directory += ''.join(map(str, args.trainingDataset)) + '/'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Logging files
    log_file_curve = open(os.path.join(log_directory, args.writer+'_all.txt'), 'w')
    log_file_pos = open(os.path.join(log_directory, args.writer + '_pos.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = os.path.join(args.save_path,'save/')
    save_directory += ''.join(map(str,args.trainingDataset)) + '/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(os.path.join(save_directory, 'social_config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a SocialModel object with the arguments
    model = SocialModel(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Initialize all variables in the graph
        sess.run(tf.initialize_all_variables())
        # writer = tf.summary.FileWriter(save_directory+args.writer)
        # writer.add_graph(sess.graph)
        # Initialize a saver that saves all the variables in the graph
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

        # summary_writer = tf.train.SummaryWriter('/tmp/lstm/logs', graph_def=sess.graph_def)
        print 'Training begin'
        best_val_loss = 100
        best_epoch = 0
        summary=0

        # For each epoch
        for e in range(args.num_epochs):
            # Assign the learning rate value for this epoch
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # Reset the data pointers in the data_loader
            data_loader.reset_batch_pointer(valid=False)

            loss_epoch = 0
            loss_epoch_pos = 0

            # For each batch
            for b in range(data_loader.num_batches):
                # Tic
                start = time.time()

                # Get the source, target and dataset data for the next batch
                # x, y are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                x, y, d = data_loader.next_batch()

                # variable to store the loss for this batch
                loss_batch = 0
                loss_batch_pos = 0

                # For each sequence in the batch
                for batch in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    x_batch, y_batch, d_batch = x[batch], y[batch], d[batch]

                    if d_batch == 0 and datasets[0] == 0:
                        dataset_data = [640, 480]
                    elif datasets[0] > 4:
                        dataset_data = [50, 50]
                    else:
                        dataset_data = [720, 576]

                    grid_batch = getSequenceGridMask(x_batch, dataset_data, args.neighborhood_size, args.grid_size)

                    # Feed the source, target data
                    feed = {model.input_data: x_batch, model.target_data: y_batch,
                            model.grid_data: grid_batch}

                    pos_loss, train_loss, _ = sess.run([model.loss, model.cost, model.train_op], feed)
                    loss_batch += train_loss
                    loss_batch_pos += pos_loss



                end = time.time()
                loss_batch = loss_batch / data_loader.batch_size
                loss_batch_pos = loss_batch_pos / data_loader.batch_size
                loss_epoch += loss_batch
                loss_epoch_pos += loss_batch_pos

                # train_cost = tf.Summary(value=[tf.Summary.Value(tag="TrainingCost", simple_value=loss_batch)])
                # writer.add_summary(train_cost, e * data_loader.num_batches + b)

                print(
                    "{}/{} (epoch {}), train_loss = {:.3f} | {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        loss_batch, loss_batch_pos, end - start))

                # Save the model if the current epoch and batch number match the frequency
                '''
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join('save', 'social_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                '''
            loss_epoch /= data_loader.num_batches
            loss_epoch_pos /= data_loader.num_batches

            print('(epoch {}), Epoch training loss = {:.3f}'.format(e, loss_epoch))

            # train_cost = tf.Summary(value=[tf.Summary.Value(tag="TrainingCost", simple_value=loss_epoch)])
            # writer.add_summary(train_cost, e * data_loader.num_batches + b)
            log_file_curve.write(str(e)+','+str(loss_epoch)+',')
            log_file_pos.write(str(e) + ',' + str(loss_epoch_pos) + ',')
            print '*****************'

            # Validation
            data_loader.reset_batch_pointer(valid=True)
            loss_epoch = 0
            loss_epoch_pos = 0
            # writer.add_graph(sess.graph)

            for b in range(data_loader.valid_num_batches):

                # Get the source, target and dataset data for the next batch
                # x, y are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                x, y, d = data_loader.next_valid_batch()

                # variable to store the loss for this batch
                loss_batch = 0
                loss_batch_pos = 0

                # For each sequence in the batch
                for batch in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    x_batch, y_batch, d_batch = x[batch], y[batch], d[batch]

                    if d_batch == 0 and datasets[0] == 0:
                        dataset_data = [640, 480]
                    elif datasets[0] > 4:
                        dataset_data = [50, 50]
                    else:
                        dataset_data = [720, 576]

                    grid_batch = getSequenceGridMask(x_batch, dataset_data, args.neighborhood_size, args.grid_size)

                    # Feed the source, target data
                    feed = {model.input_data: x_batch, model.target_data: y_batch,
                            model.grid_data: grid_batch}

                    train_loss = sess.run(model.cost, feed)
                    loss_batch += train_loss
                    loss_batch_pos += pos_loss

                loss_batch = loss_batch / data_loader.batch_size
                loss_batch_pos = loss_batch_pos / data_loader.batch_size
                loss_epoch += loss_batch
                loss_epoch_pos += loss_batch_pos

                # test_cost = tf.Summary(value=[tf.Summary.Value(tag="TestCost", simple_value=loss_batch)])
                # writer.add_summary(test_cost, e * data_loader.num_batches + b)
                print(
                    "{}/{} (epoch {}), Valid_loss = {:.3f} | {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.valid_num_batches,
                        e,
                        loss_batch, loss_batch_pos, end - start))

            loss_epoch /= data_loader.valid_num_batches
            loss_epoch_pos /= data_loader.valid_num_batches
            # test_cost = tf.Summary(value=[tf.Summary.Value(tag="TestCost", simple_value=loss_epoch)])
            # writer.add_summary(test_cost, e * data_loader.num_batches + b)

            # Update best validation loss until now
            if loss_epoch_pos < best_val_loss:
                best_val_loss = loss_epoch_pos
                best_epoch = e

            print('(epoch {}), Epoch validation loss = {:.3f} | {:.3f}'.format(e, loss_epoch, loss_epoch_pos))
            print 'Best epoch', best_epoch, 'Best validation loss', best_val_loss
            log_file_curve.write(str(loss_epoch)+'\n')
            log_file_curve.write(str(loss_epoch_pos)+'\n')

            print '*****************'

            # Save the model after each epoch
            if e % args.save_every == 0 and (e > 0):
                print 'Saving model'
                checkpoint_path = os.path.join(save_directory, 'social_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e)
                print("model saved to {}".format(checkpoint_path))

        print 'Best epoch', best_epoch, 'Best validation loss', best_val_loss
        log_file.write(str(best_epoch)+','+str(best_val_loss))

        # tf.summary.merge_all()

        # CLose logging files
        log_file.close()
        log_file_curve.close()


if __name__ == '__main__':
    main()
