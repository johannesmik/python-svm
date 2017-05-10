__author__ = 'johannes'

import cProfile
import sys
import numpy as np
from preprocess import *
from plot import *
from svm import *


def generate_data(samples=100, classes=2):
    """
        Generates two dimensional data

    Samples - samples of each class to create
    Classes - different classes
    Returns: DataArray, LabelVector
    """
    dataArr = []
    labelArr = []
    for c in range(classes):
        x = np.random.randint(-10, 10)
        y = np.random.randint(-10, 10)

        for i in range(samples):
            x1 = np.random.normal()
            y1 = np.random.normal()
            dataArr.append([x1 + x, y1 + y])
            labelArr.append(c)

    return np.array(dataArr), np.array(labelArr)


def read_data(file_in):
    datas = []
    labels = []
    for line in open(file_in):
        data = line.strip().split()[:-1]
        data = map(float, data)
        datas.append(data)
        label = line.strip().split()[-1]
        labels.append(label)
    datas = np.array(datas)
    labels = np.array(labels)
    return datas, labels


def cross_validate(data, labels, k=2, sigma=1.0, c=1.0):
    """
        k-cross validation
    """

    ids = np.random.randint(k, size=len(data))

    error = 0.0

    # Train and test k models
    for i in range(k):
        model = SvmMulticlassOneOne(data[np.where(ids != i)], labels[np.where(ids != i)], sigma=sigma, c=c)
        model.train()
        error += test(model, data[np.where(ids == i)], labels[np.where(ids == i)])
        print "Cross-Validation %i th fold %f acc.error" % (i, error)
        print
        # return average error
    return error / float(k)


def test(model, data, labels):
    """ Use model to test the data, compare with labels
        Returns error rate and confusion matrix
    """
    classified_cor = 0
    classified_tot = 0

    # Build the confusion matrix
    uniquelabels = model.uniquelabels.tolist()
    confusion = np.zeros((len(uniquelabels), len(uniquelabels)))

    # Classify each datum with model and compare to label
    for date, label in zip(data, labels):
        prediction = model.predict(date)
        if prediction == label:
            classified_cor += 1

        classified_tot += 1

        # Update confusion matrix
        i = uniquelabels.index(label)
        j = uniquelabels.index(prediction)
        confusion[i, j] += 1

    error = 1 - classified_cor / float(classified_tot)
    print "Error rate", error
    return error, confusion


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        option = sys.argv[1]
    else:
        # Choose option by hand
        option = "2d_examples"
        #option = "2d_examples_oneall"
        #option = "find_sigma"
        #option = "find_c"
        #option = "letter_data"
        #option = "letter_data_oneall"

    # Call preprocess
    preprocess("data/letter-recognition.data", "data/letter-recognition-training.data",
               "data/letter-recognition-validation.data")

    if option == "letter_data":

        data_train, label_train = read_data("data/letter-recognition-training.data")
        model = SvmMulticlassOneOne(data_train, label_train, kernel="gaussian", sigma=3.1622, c=1.0)
        print "Warning: training takes some time"
        cProfile.run('model.train()')

        data_test, label_test = read_data("data/letter-recognition-validation.data")

        print "Testing the model (takes some time):"
        error, confusion = test(model, data_test, label_test)
        plot_confusion(confusion, model.uniquelabels.tolist())
        print model.stats()
        plt.show()

    elif option == "letter_data_oneall":
        data_train, label_train = read_data("data/letter-recognition-training.data")

        # Fixme: currently only trains on the first 7000 examples
        data_train = data_train[:8000]
        label_train = label_train[:8000]

        model = SvmMulticlassOneAll(data_train, label_train, kernel="gaussian", sigma=3.1622, c=1.0)
        print "Warning: training takes some time"
        cProfile.run('model.train()')

        data_test, label_test = read_data("data/letter-recognition-validation.data")

        print "Testing the model"
        error, confusion = test(model, data_test, label_test)
        plot_confusion(confusion, model.uniquelabels.tolist())
        print model.stats()
        plt.show()

    elif option == "2d_examples":

        samples = 200
        classes = 4

        dataArray, labelArray = generate_data(samples, classes)
        model = SvmMulticlassOneOne(dataArray, labelArray, c=1, kernel="linear")
        cProfile.run('model.train()')

        # Fixme: Currently tested on training data
        error, confusion = test(model, dataArray, labelArray)

        plot_onevsone(model)
        plot_confusion(confusion, model.uniquelabels.tolist())
        print "Statistics: ",  model.stats()
        plt.show()

    elif option == "2d_examples_oneall":

        samples = 200
        classes = 4

        dataArray, labelArray = generate_data(samples, classes)
        model = SvmMulticlassOneAll(dataArray, labelArray, c=1, kernel="linear")
        model.train()

        # Fixme: Currently tested on training data
        error, confusion = test(model, dataArray, labelArray)

        plot_onevsall(model)
        plot_confusion(confusion, model.uniquelabels.tolist())
        print "Statistics: ",  model.stats()
        plt.show()

    elif option == "find_sigma":

        data, label = read_data("data/letter-recognition-training.data")

        sigma_start = 1.0
        sigma_end = 10.0
        values = 30

        # Define training and testing sets
        data_train = data[:4000, ]
        label_train = label[:4000, ]
        data_test = data[4000:4250, ]
        label_test = label[4000:4250, ]

        error = []
        for sigma in np.linspace(sigma_start, sigma_end, values):
            model = SvmMulticlassOneOne(data_train, label_train, kernel="gaussian", sigma=sigma, c=1.0)
            model.train()
            e, c = test(model, data_test, label_test)
            error.append(e)

        for sigma, e in zip(np.linspace(sigma_start, sigma_end, values), error):
            print "sigma", sigma, "error", e

        sigmas = np.linspace(sigma_start, sigma_end, values).tolist()

        plot_sigma(sigmas, error)
        plt.show()

    elif option == "find_c":
        data, label = read_data("data/letter-recognition-training.data")

        c_list = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

        # Define training and testing sets
        data_train = data[:4000, ]
        label_train = label[:4000, ]
        data_test = data[4000:4250, ]
        label_test = label[4000:4250, ]

        error = []
        for c1 in c_list:
            model = SvmMulticlassOneOne(data_train, label_train, kernel="gaussian", sigma=2.915, c=c1)
            model.train()
            e, confusion = test(model, data_test, label_test)
            error.append(e)

        for c1, e in zip(c_list, error):
            print "C", c1, "error", e

        plot_c(c_list, error)
        plt.show()
