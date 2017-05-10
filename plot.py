__author__ = 'johannes'

import numpy as np
import matplotlib.pyplot as plt
import svm


def plot_confusion(confusion, labels):
    """ Plots the confusion matrix

        confusion: the (n, n) confusion matrix
        labels: (sorted) list of labels
    """

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(confusion, cmap=plt.cm.BuPu,
                    interpolation='nearest')

    width, height = confusion.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(int(confusion[x, y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    plt.xticks(range(width), labels)
    plt.yticks(range(height), labels)
    fig.canvas.set_window_title("Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')


def plot_onevsone(model):
    " Plot the first two dimensions of a OneVsOne SVM "

    # Only works for One-vs-One models and kernel must be linear
    if not isinstance(model, svm.SvmMulticlassOneOne):
        return 0
    if model.kernel != "linear":
        return 0

    class_shapes = ['o', '^', 'v', 'h', '+', 'D']
    colors = ['b', 'r', 'y', 'g', 'y', 'k']

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    # Iterate over all Svms and plot the line
    for c1 in range(len(model.uniquelabels) - 1):
        for c2 in range(c1 + 1, len(model.uniquelabels)):
            w = model.svms[model.uniquelabels[c1]][model.uniquelabels[c2]].w
            b = model.svms[model.uniquelabels[c1]][model.uniquelabels[c2]].b

            x1 = np.linspace(-13, 13, 50)
            x2 = (-w[0] * x1 + b) / w[1]
            ax.plot(x1, x2, '-', label="%s" % model.svms[model.uniquelabels[c1]][model.uniquelabels[c2]])

    for c1 in range(len(model.uniquelabels)):
        m = class_shapes[c1%6]
        c = colors[c1%6]
        ax.scatter(model.data[np.where(model.label == model.uniquelabels[c1]), 0],
                   model.data[np.where(model.label == model.uniquelabels[c1]), 1], marker=m, c=c, alpha=0.5,
                   label=str(c1))
    plt.legend()
    plt.ylim(-13, 13)
    fig.canvas.set_window_title("Linear SVMS - One Vs One")


def plot_onevsall(model):
    " Plot the first two dimensions of a OneVsOne SVM "

    # Only works for One-vs-One models and kernel must be linear
    if not isinstance(model, svm.SvmMulticlassOneAll):
        return 0
    if model.kernel != "linear":
        return 0

    class_shapes = ['o', '^', 'v', 'h', '+', 'D']
    colors = ['b', 'r', 'y', 'g', 'y', 'k']

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    # Iterate over all Svms and plot the line
    for c1 in range(len(model.uniquelabels)):
        w = model.svms[model.uniquelabels[c1]].w
        b = model.svms[model.uniquelabels[c1]].b

        x1 = np.linspace(-13, 13, 50)
        x2 = (-w[0] * x1 + b) / w[1]
        ax.plot(x1, x2, '-', label="%s" % model.svms[model.uniquelabels[c1]])

    for c1 in range(len(model.uniquelabels)):
        m = class_shapes[c1]
        c = colors[c1]
        ax.scatter(model.data[np.where(model.label == model.uniquelabels[c1]), 0],
                   model.data[np.where(model.label == model.uniquelabels[c1]), 1], marker=m, c=c, alpha=0.5,
                   label=str(c1))
    plt.legend()
    plt.ylim(-13, 13)
    fig.canvas.set_window_title("Linear SVMS - One vs All")


def plot_sigma(sigma, error):
    """
     Sigma: list
     Error: list
    """
    sigma = np.array(sigma)
    error = np.array(error)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    #ax.set_aspect(1)

    ax.scatter(sigma, error)
    plt.ylabel('Error')
    plt.xlabel('sigma')


def plot_c(c, error):
    """
     c: list of c-values
     Error: list
    """
    c = np.array(c)
    error = np.array(error)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    #ax.set_aspect(1)

    ax.scatter(c, error)
    plt.ylabel('Error')
    plt.xlabel('C')


