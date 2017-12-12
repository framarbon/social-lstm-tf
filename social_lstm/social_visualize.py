'''
Script to help visualize the results of the trained model

Author : Anirudh Vemula
Date : 10th November 2016
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_trajectories(true_trajs, pred_trajs, obs_length, name):
    '''
    Function that plots the true trajectories and the
    trajectories predicted by the model alongside
    params:
    true_trajs : numpy matrix with points of the true trajectories
    pred_trajs : numpy matrix with points of the predicted trajectories
    Both parameters are of shape traj_length x maxNumPeds x 3
    obs_length : Length of observed trajectory
    name: Name of the plot
    '''
    traj_length, maxNumPeds, _ = true_trajs.shape

    # Initialize figure
    plt.figure()

    # Load the background
    im = plt.imread('plot/background.png')
    implot = plt.imshow(im)
    width = im.shape[0]
    height = im.shape[1]
    print 'width '+str(width)
    print 'height '+str(height)

    limits_x = []
    limits_y = []

    # width = 1
    # height = 1

    traj_data = {}
    # For each frame/each point in all trajectories
    for i in range(traj_length):
        pred_pos = pred_trajs[i, :]
        true_pos = true_trajs[i, :]

        # For each pedestrian
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Not a ped
                continue
            elif pred_pos[j, 0] == 0:
                # Not a ped
                continue
            else:
                # If he is a ped
                if true_pos[j, 1] > 1 or true_pos[j, 1] < -1:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < -1:
                    continue

                if (j not in traj_data) and i < obs_length:
                    traj_data[j] = [[], []]

                if j in traj_data:
                    traj_data[j][0].append(true_pos[j, 1:3])
                    traj_data[j][1].append(pred_pos[j, 1:3])

    for index, j in enumerate(traj_data):
        # c = np.random.rand(3, 1)
        c = [1, 0, 0]
        c2 = [0, 0, 1]
        c3 = [0, 1, 0]
        c4 = [1, 0, 1]
        true_traj_ped = traj_data[j][0]  # List of [x,y] elements
        pred_traj_ped = traj_data[j][1]

        true_x = [(1.+p[0])*height/2. for p in true_traj_ped]
        true_y = [(1.+p[1])*width/2.  for p in true_traj_ped]
        pred_x = [(1.+p[0])*height/2. for p in pred_traj_ped]
        pred_y = [(1.+p[1])*width/2.  for p in pred_traj_ped]

        if index == 0:
            c2= c4
            c = c3
            limits_x=pred_x
            limits_y=pred_y

        plt.plot(true_x, true_y, color=c, linestyle='solid', marker='o')
        plt.plot(pred_x, pred_y, color=c2, linestyle='dashed', marker='x')


    plt.xlim(min(limits_x[0],limits_x[-1])-0.12*width, max(limits_x[0],limits_x[-1])+0.12*width)
    plt.ylim(min(limits_y[0],limits_y[-1])-0.12*height, max(limits_y[0],limits_y[-1])+0.12*height)
    plt.show()
    # plt.savefig('plot/'+name+'.png')
    plt.gcf().clear()
    plt.close()


def main():
    '''
    Main function
    '''
    f = open('save/social_results.pkl', 'rb')
    results = pickle.load(f)

    for i in range(len(results)):
        print i
        name = 'sequence' + str(i)
        plot_trajectories(results[i][0], results[i][1], results[i][2], name)



if __name__ == '__main__':
    main()
