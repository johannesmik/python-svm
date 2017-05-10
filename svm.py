__author__ = 'johannes'

"""
The implementation of the SMO algorithm is based on John C. Platt's description:
"Fast Training of Support Vector Machines using Sequential Minimal Optimization" (1999)
"""

import time
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from kernel import *


class SvmMulticlassOneOne(object):
    def __init__(self, data, label, kernel='gaussian', sigma=1.0, c=1.0, tol=0.01):
        self.data = data
        self.label = label
        self.kernel = kernel
        self.sigma = sigma
        self.c = c
        self.tol = tol
        self.uniquelabels = np.unique(self.label)
        self.svms = {}
        self.create_svms()
        self.trainingtime = 0

    def stats(self):
        " Return a dictionary with some stats about this model "
        if self.trainingtime == 0:
            # Model has not been trained yet
            return  {'classification': 'one-vs-one'}

        svms = 0
        support_vectors = 0
        for c1 in self.svms:
            for c2 in self.svms[c1]:
                svms += 1
                support_vectors += self.svms[c1][c2].number_supportvectors()

        return {'classification': 'one-vs-one', 'kernel': self.kernel, 'sigma': self.sigma, 'c': self.c, 'tol': self.tol,
                'training-time': self.trainingtime, 'svms': svms, 'support-vectors': support_vectors}

    def prepare_data(self, firstlabel, secondlabel):

        label_tmp = np.zeros(len(self.label))
        label_tmp[np.where(self.label == firstlabel)] = 1
        label_tmp[np.where(self.label == secondlabel)] = -1

        return self.data[np.where(label_tmp != 0)], label_tmp[np.where(label_tmp != 0)]

    def create_svms(self):
        # Iterate over all classes
        for c1 in range(len(self.uniquelabels) - 1):
            svms = {}
            for c2 in range(c1 + 1, len(self.uniquelabels)):
                label1 = self.uniquelabels[c1]
                label2 = self.uniquelabels[c2]
                t_data, t_label = self.prepare_data(label1, label2)
                svm = Svm(t_data, t_label, c=self.c, sigma=self.sigma, kernel=self.kernel, tol=self.tol,
                          name="%s<->%s" % (label1, label2))
                svms[self.uniquelabels[c2]] = svm
            self.svms[self.uniquelabels[c1]] = svms

    def train(self):
        start = time.time()
        for c1 in self.svms:
            for c2 in self.svms[c1]:
                self.svms[c1][c2].train()
        stop = time.time()
        self.trainingtime = stop-start
        print "Trained all SVMs"

    def predict(self, x):
        score = {}
        for c in self.uniquelabels:
            score[c] = 0

        for c1 in range(len(self.uniquelabels) - 1):
            for c2 in range(c1 + 1, len(self.uniquelabels)):
                prediction = self.svms[self.uniquelabels[c1]][self.uniquelabels[c2]].predict(x)
                if prediction == 1:
                    score[self.uniquelabels[c1]] += 1
                else:
                    score[self.uniquelabels[c2]] += 1

        return sorted(score, key=score.get, reverse=True)[0]


class SvmMulticlassOneAll(object):
    def __init__(self, data, label, kernel='gaussian', sigma=1.0, c=1.0, tol=0.01):
        self.data = data
        self.label = label
        self.kernel = kernel
        self.sigma = sigma
        self.c = c
        self.tol = tol
        self.uniquelabels = np.unique(self.label)
        self.trainingtime = 0
        self.svms = {}
        self.create_svms()

    def stats(self):
        " Return a dictionary with some stats about this model "
        if self.trainingtime == 0:
            # Model has not been trained yet
            return {'classification': 'one-vs-all'}

        svms = 0
        support_vectors = 0
        for c in self.svms:
            svms += 1
            support_vectors += self.svms[c].number_supportvectors()

        return {'kernel': self.kernel, 'classification': 'one-vs-all', 'sigma': self.sigma, 'c': self.c, 'tol': self.tol,
                'training-time': self.trainingtime, 'svms': svms, 'support-vectors': support_vectors}

    def prepare_labels(self, firstlabel):

        label_tmp = np.zeros(len(self.label))
        label_tmp[np.where(self.label == firstlabel)] = 1
        label_tmp[np.where(self.label != firstlabel)] = -1

        #self.label = label_tmp[np.where(label_tmp != 0)]
        #self.data = self.data[np.where(label_tmp != 0)]

        return label_tmp

    def create_svms(self):
        # Iterate over all classes
        for c in self.uniquelabels:
            t_label = self.prepare_labels(c)
            svm = Svm(self.data, t_label, c=self.c, tol=self.tol, sigma=self.sigma, kernel=self.kernel, name="%s - All" % c)
            self.svms[c] = svm

    def train(self):
        start = time.time()
        for c in self.svms:
            self.svms[c].train()
        stop = time.time()
        self.trainingtime = stop-start
        print "Trained all SVMs"

    def predict(self, x):
        score = {}
        for l in self.uniquelabels:
            score[l] = self.svms[l].classify(x)

        return max(score, key=score.get)


class Svm(object):
    def __init__(self, data, label, c, tol, sigma, kernel, name):
        self.data = data
        self.label = label
        self.c = c
        self.tol = tol
        if kernel == 'gaussian':
            self.kernel = lambda x, y: gaussian_kernel(x, y, sigma=sigma)
        else:
            self.kernel = linear_kernel
        self.name = name

        self.n, self.d = self.data.shape
        self.w = np.zeros(self.d)
        self.b = 0 # Threshold
        self.alpha = np.zeros(self.n)
        self.error_cache = np.zeros(self.n)
        self.error_cache_valid = np.zeros(self.n)

    def __str__(self):
        return "SVM: %s" % self.name

    def number_supportvectors(self):
        " Return number of non-zero alphas "
        return len(np.where(self.alpha != 0)[0])

    def predict(self, x):
        if self.classify(x) >= 0:
            return 1
        else:
            return -1

    def classify(self, x):
        """ Classify data vector x, depending on current alphas"""
        prediction = np.dot((self.alpha * self.label), self.kernel(self.data, x)) - self.b
        return prediction

    def update_weights(self):
        self.w = np.zeros(self.d)
        for i in range(self.n):
            self.w += self.alpha[i] * self.label[i] * self.data[i]

    @staticmethod
    def clip(alpha, l, h):
        if alpha >= h:
            return h
        elif alpha > l:
            return alpha
        else:
            return l

    def solve_two(self, i, j):
        """ Solve for the two Lagrange multipliers alpha[i] and alpha[j]"""
        # As described in section 12.2.1
        #print
        #print "solve two", i, j
        if i == j:
            return 0

        alpha_i = self.alpha[i]
        alpha_j = self.alpha[j]
        label_i = self.label[i]
        label_j = self.label[j]

        ei = self.get_error(i)
        ej = self.get_error(j)

        s = label_i * label_j

        kii = self.kernel(self.data[i], self.data[i])
        kij = self.kernel(self.data[i], self.data[j])
        kjj = self.kernel(self.data[j], self.data[j])

        # Compute L and H
        if label_i != label_j:
            l = max(0, alpha_j - alpha_i)
            h = min(self.c, self.c + alpha_j - alpha_i)
        else:
            l = max(0, alpha_i + alpha_j - self.c)
            h = min(self.c, alpha_i + alpha_j)

        if l == h:
            # This means that at least one of the alphas is 0 or C
            #print "l == h", L, H, label_i, label_j
            return 0

        # Second derivative of the objective function
        eta = 2.0 * kij - kii - kjj

        if eta > 0.0:
            return 0
        else:
            alpha_j_new = alpha_j - (label_j * (ei - ej)) / eta
            alpha_j_new = self.clip(alpha_j_new, l, h)

        if alpha_j_new < 1e-8:
            alpha_j_new = 0
        elif alpha_j_new > self.c - 1e-8:
            alpha_j_new = self.c

        if abs(alpha_j_new - alpha_j) < 1e-3:
            return 0

        alpha_i_new = alpha_i + s * (alpha_j - alpha_j_new)

        # Update threshold b
        b1 = ei + label_i * (alpha_i_new - alpha_i) * kii + \
             label_j * (alpha_j_new - alpha_j) * kij + self.b

        b2 = ej + label_i * (alpha_i_new - alpha_i) * kij + \
             label_j * (alpha_j_new - alpha_j) * kjj + self.b

        if 0 < alpha_i_new < self.c:
            self.b = b1
        elif 0 < alpha_j_new < self.c:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

        # Update alphas
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new

        # Update error cache using new Lagrange multipliers
        self.error_cache[i] = self.classify(self.data[i]) - label_i
        self.error_cache[j] = self.classify(self.data[j]) - label_j
        self.error_cache_valid[i] = 1
        self.error_cache_valid[j] = 1

        return 1

    def get_error(self, j):

        if self.error_cache_valid[j] == 1:
            return self.error_cache[j]
        else:
            return self.classify(self.data[j]) - self.label[j]


    def examine_example(self, j):
        label_j = self.label[j]
        alpha_j = self.alpha[j]
        # Check in error_cache
        ej = self.get_error(j)
        #ej = self.classify(self.data[j]) - label_j
        rj = ej * label_j
        if (rj < - self.tol and alpha_j < self.c) or (rj > self.tol and alpha_j > 0):
            if len((self.alpha > 0) & (self.alpha < self.c)) > 1:
                if ej > 0:
                    i = self.error_cache.argmin()
                else:
                    i = self.error_cache.argmax()

                if self.solve_two(i, j):
                    return 1
                    # Loop over all non-zero and non-C alpha, starting at random point
            for i in np.where((self.alpha != 0) & (self.alpha != self.c))[0]:
                if self.solve_two(i, j):
                    return 1
                    # Loop over all possible i1, starting at a random point
            for i in np.random.permutation(np.arange(0, self.n)):
                if self.solve_two(i, j):
                    return 1
        return 0

    def train(self):
        print "start training...", self
        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                # Loop over all training examples
                for j in range(self.n):
                    # examine example j
                    num_changed += self.examine_example(j)
            else:
                # Loop over all examples where alpha is neither 0 nor c
                for j in np.where((self.alpha != 0) & (self.alpha != self.c))[0]:
                    num_changed += self.examine_example(j)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
        print "training finished"
        # Update weights
        self.update_weights()

