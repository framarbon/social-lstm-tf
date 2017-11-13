import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import argparse
from cycler import cycler
import itertools

filenames = []
folders = []

def plot_learning_curve(log_path, num_epochs, title="Benchmark"):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    log_path : Path were the log files are stored

    title : string
        Title for the chart.
    """
    directories = [name for name in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, name))]
    if len(directories) == 0:
        num_samples = 1
    else:
        num_samples = len(directories)
    samples = range(num_samples)
    train_scores = []
    test_scores = []
    names = []

    di = 0
    for root, dirs, files in os.walk(log_path):
        sfiles = sorted(files)
        fi = 0
        for file in sfiles:
            if file.endswith(".txt") and file.endswith("lr=", 0, 3):
                if len(train_scores) <= fi:
                    train_scores.append(np.zeros([num_epochs, num_samples]))
                    test_scores.append(np.zeros([num_epochs, num_samples]))
                    names.append(file.split('_')[0])
                with open(os.path.join(root, file), 'rb') as f:
                    # f.readline()  # skip the header
                    index = 0
                    for row in f:
                        _, train, test = row.split(',')
                        train_scores[fi][index, samples[di]] = float(train)
                        test_scores[fi][index, samples[di]] = float(test)
                        index += 1
                fi += 1
        if fi > 0:
            di += 1

    train_scores_means = np.mean(train_scores, axis=2)
    train_scores_stds = np.std(train_scores, axis=2)
    test_scores_means = np.mean(test_scores, axis=2)
    test_scores_stds = np.std(test_scores, axis=2)

    num_curves = 3
    test_min = np.min(test_scores_means, axis=1)
    print test_min
    zipp = zip(test_min, train_scores_means, train_scores_stds, test_scores_means, test_scores_stds, names)
    zipp.sort(key=lambda t: t[0])
    # plt.grid()

    clist = mpl.rcParams['axes.color_cycle'] = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue'] #['r', 'g', 'b', 'y']
    cgen0 = itertools.cycle(clist)
    cgen1 = itertools.cycle(clist)
    # Alternately, we could use rc:
    # mpl.rc('axes', color_cycle=['r','k','c'])
    fig, axes = plt.subplots(nrows=2)

    k=0
    for i, (_, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, name) in enumerate(zipp):
        if i< 0:
            continue
        axes[0].fill_between(range(num_epochs), train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color=cgen0.next())
        axes[1].fill_between(range(num_epochs), test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color=cgen1.next())
        axes[0].plot(range(num_epochs), train_scores_mean, 'o-',  label=name)
        axes[1].plot(range(num_epochs), test_scores_mean, 'o-', label=name)
        k += 1
        if k == num_curves:
            break

    handles, labels = axes[1].get_legend_handles_labels()
    lgd = axes[1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    axes[0].grid('on')
    axes[1].grid('on')
    # plt.legend(loc="best")
    return plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default="",
                        help='Path of saved log files')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Num of epochs during training')
    args = parser.parse_args()

    plot_learning_curve(args.log_path, args.num_epochs)


    plt.show()