'''
Script to help visualize the results of the trained model

Author : Anirudh Vemula
Date : 10th November 2016
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import seaborn as sns


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
    # fig = plt.figure()
    # ax1 = fig.add_subplot(2, 1, 1)
    # ax2 = fig.add_subplot(2, 1, 2)

    # f, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True)
    fig1 = plt.figure()
    # fig2 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    # ax2 = fig2.add_subplot(1, 1, 1)

    # Load the background
    im = plt.imread('plot/background.png')
    width = im.shape[0]
    height = im.shape[1]

    limits_x = []
    limits_y = []

    ped_x  = []
    ped_y = []

    traj_data = {}
    pred_data = {}
    obs_data = {}
    obs_samples_lem=0
    # For each frame/each point in all trajectories
    for i in range(traj_length):
        # pred_pos = pred_[i, :]
        true_pos = true_trajs[i, :]

        # For each pedestrian
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Not a ped
                continue
            elif pred_trajs[0][i][j][0] == 0:
                # Not a ped
                continue
            else:
                # If he is a ped
                if true_pos[j, 1] > 1 or true_pos[j, 1] < -1:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < -1:
                    continue

                if (j not in traj_data) and i < obs_length:
                    traj_data[j] = []
                    pred_data[j] = []
                    obs_data[j] = []

                if j in traj_data:
                    traj_data[j].append(list(true_pos[j, 1:3]))
                    for index, sample in enumerate(pred_trajs):
                        if i >= obs_length:
                            pred_data[j].append(list(sample[i][j,1:3]))
                        else:
                            obs_data[j].append(list(sample[i][j, 1:3]))
                        # traj_data[j][1].append(pred_pos[j, 1:3])

    for index, j in enumerate(traj_data):
        # c = np.random.rand(3, 1)
        c = [1, 0, 0]
        c2 = [1, 0.5, 0]
        c4 = [0, 0.8, 0]
        c3 = [0, 0.7, 0.7]
        true_traj_ped = traj_data[j]  # List of [x,y] elements
        pred_traj_ped = pred_data[j]

        true_x = [(1.+p[0])*height/2. for p in true_traj_ped]
        true_y = [(1.+p[1])*width/2.  for p in true_traj_ped]
        pred_x = [(1.+p[0])*height/2. for p in pred_traj_ped]
        pred_y = [(1.+p[1])*width/2.  for p in pred_traj_ped]

        label1 = 'Pedestrian observed states'
        label2 = 'Pedestrian no-observed states'

        if index == 0:
            c2= c4
            c = c3
            limits_x = [(1. + p[0]) * height / 2. for p in obs_data[j]]
            limits_y = [(1. + p[1]) * width / 2. for p in obs_data[j]]
            obs_samples_lem = len(limits_x)
            limits_x.extend(pred_x)
            limits_y.extend(pred_y)
            label1 = 'Robot observed states'
            label2 = 'Robot no-observed states'

        if obs_length <= len(true_x):
            ax1.plot(true_x[obs_length-1:len(true_x)], true_y[obs_length-1:len(true_y)], color=c2, linestyle='solid', marker='o', label=label2)
            # ax2.plot(true_x[obs_length-1:len(true_x)], true_y[obs_length-1:len(true_y)], color=c2, linestyle='solid', marker='o', label=label2)

        ax1.plot(true_x[0:min(obs_length, len(true_x))], true_y[0:min(obs_length,len(true_y))], color=c, linestyle='solid', marker='o', label=label1)
        # ax2.plot(true_x[0:min(obs_length, len(true_x))], true_y[0:min(obs_length,len(true_y))], color=c, linestyle='solid', marker='o', label=label1)


        # plt.plot(pred_x, pred_y, color=c2, linestyle='dashed', marker='x')
        if index > 0:
            ped_x.extend(pred_x)
            ped_y.extend(pred_y)
            # plt.hist2d(pred_x, pred_y, bins=40, norm=LogNorm())

    sns.kdeplot(ped_x, ped_y, shade=True, ax=ax1, shade_lowest=False, cmap="Blues")
    sns.kdeplot(limits_x[obs_samples_lem:len(limits_x)], limits_y[obs_samples_lem:len(limits_x)], shade=True, ax=ax1, shade_lowest=False)
    ax1.imshow(im)
    # ax2.imshow(im)

    low_x_lim = min(limits_x[0],limits_x[-1])-0.12*width
    high_x_lim = max(limits_x[0],limits_x[-1])+0.12*width
    low_y_lim = min(limits_y[0],limits_y[-1])-0.12*height
    high_y_lim = max(limits_y[0],limits_y[-1])+0.12*height

    leg1 = ax1.legend(loc='lower right', shadow=True)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # leg2 = ax2.legend(loc='lower right', shadow=True)

    # leg1 = ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # leg2 = ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax1.set_xlim(low_x_lim, high_x_lim)
    ax1.set_ylim(low_y_lim, high_y_lim)
    # ax2.set_xlim(low_x_lim, high_x_lim)
    # ax2.set_ylim(low_y_lim, high_y_lim)
    # plt.ylim(round(min(limits_y[0],limits_y[-1])-0.12*height), round(max(limits_y[0],limits_y[-1])+0.12*height))
    # plt.xlim(min(limits_x[0],limits_x[-1])-0.12*width, max(limits_x[0],limits_x[-1])+0.12*width)
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

        name = 'sequence' + str(i)
        plot_trajectories(results[i][0][0], results[i][1:-1], results[i][0][1], name)



if __name__ == '__main__':
    main()
