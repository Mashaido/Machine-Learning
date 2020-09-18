"""
The Perceptron algorithm is the simplest type of artificial neural network.

implementing a perceptron algorithm (from scratch) and testing it on the PerceptronData dataset using ten fold
cross validation

• PerceptronData: This is a binary classification dataset consisting of four features and the classes are linearly
separable

For both the perceptron and dual perceptron be sure to set an upper limit on the number of iterations

Use k-fold cross validation to estimate the performance of the learned model on unseen data
i.e construct and evaluate k models and estimate the performance as the mean model error. use classification accuracy
to evaluate each model. provide these behaviors in the cross_validation_split(), accuracy_metric() and
evaluate_algorithm() helper functions.



"""
import statistics
from random import randrange, seed

import numpy
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, accuracy_score


# method to load csv file
def load_csv(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    return dataset


# ____________________________________________________________________________________________________________________

# NORMALIZATION

# ____________________________________________________________________________________________________________________

'''
as a pre-processing step, I normalize continuous features

the correct strategy for normalization is to normalize the data based on training data and then use the same
information to normalize test data

unlike ridge regression do not normalize the labels which are discrete and represent the class labels

the goal of normalization is to make every data point have the same scale so each 
feature is equally important and
no one feature dominates the others

Z-score is a strategy of normalizing data that avoids the outlier issue that comes with min-max normalization
(value−μ)/ σ

normalizing training data, and then using the normalization parameters to normalize my test data and then
estimate the accuracy/error
'''


# method to normalize the features in the training (then test) data using z-score normalization
def z_score_normalize(dataset, mean, standard_deviation):
    for row in dataset:
        for col in range(len(row)):
            if row[col] is not None:
                row[col] = (row[col] - mean[col]) / standard_deviation[col]


# ____________________________________________________________________________________________________________________

# CALCULATING MEAN AND STANDARD DEVIATION

# ____________________________________________________________________________________________________________________

# method to get the mean and standard deviation of each column
def mean_standard_deviation(dataset):
    mean = []
    standard_deviation = []
    for i in range(len(dataset[0])):
        col = [row[i] for row in dataset]
        mean.append(statistics.mean(col))
        standard_deviation.append(statistics.stdev(col))
    return mean, standard_deviation


# ____________________________________________________________________________________________________________________

# ESTIMATING COEFFICIENTS

# ____________________________________________________________________________________________________________________

# method to finding a set of weights for the model which minimizes my cost function
def gradient_descent(train, y_train, learning_rate, number_of_iterations, tolerance, boolean):
    # note that learning rate controls how much the weight/coeff and y-intercept values changes with each step
    # i.e used to limit the amount each weight is corrected each time it is updated

    # number_of_iterations is the number of times to run through the training data while updating the weight

    train = numpy.array(train)

    # add a default bias term of 1 (x_0) to features i.e add 1 as a feature column
    add_default_bias_term = np.ones((train.shape[0], 1))
    train = np.hstack((add_default_bias_term, train))

    # initialize weights for the gradient descent algorithm to all zeros i.e: w = [0, 0, . . . , 0]^T
    weights = numpy.zeros(train.shape[1])

    # keep track of iterations i.e algorithm runs for number_of_iterations
    # i.e loop over each number_of_iterations
    for step in range(number_of_iterations):
        sum_error = 0
        index = 0
        # loop over each row in the training data for a number_of_iteration
        for row in train:
            predictions = predict(row, weights)
            output_error = abs(y_train[index] - predictions)
            # if there's a prediction error then adjust the weights accordingly
            if predictions * y_train[index] <= 0:
                # loop over each weight and update it for a row in a number_of_iteration
                for i in range(len(weights)):
                    weights[i] += learning_rate * y_train[index] * train[index][i]
            sum_error += output_error
            index += 1

        # convergence criteria for the gradient descent algorithm
        print(sum_error)
        if sum_error == 0:
            break

    return weights, train


# ____________________________________________________________________________________________________________________

# MAKING PREDICTIONS

# ____________________________________________________________________________________________________________________

# method that uses the estimated weights/coefficients to make predictions with a perceptron algorithm model
# i.e implementing a single preceptron algorithm
def perceptron_algorithm(train, test, y_train, y_test, number_of_iterations, tolerance, learning_rate, boolean):
    pred_train = []
    pred_test = []

    # get weights from gradient_descent()
    weights, train = gradient_descent(train, y_train, learning_rate, number_of_iterations, tolerance, boolean)
    print("-" * 50)
    for row in train:
        pred_train.append(predict(row, weights))

    # calculate the training precision, recall and accuracy
    train_precision = precision_score(y_train, pred_train)
    train_recall = recall_score(y_train, pred_train)
    train_accuracy = accuracy_score(y_train, pred_train)

    # create test data
    test = numpy.array(test)

    # add a default bias term of 1 (x_0) to test features
    add_default_bias_term = np.ones((test.shape[0], 1))
    test = np.hstack((add_default_bias_term, test))

    for row in test:
        pred_test.append(predict(row, weights))

    # calculate the testing precision, recall and accuracy
    test_precision = precision_score(y_test, pred_test)
    test_recall = recall_score(y_test, pred_test)
    test_accuracy = accuracy_score(y_test, pred_test)

    return train_precision, train_recall, train_accuracy, test_precision, test_recall, test_accuracy


# helper function to make predictions given dataset and weights
def predict(row, weights):
    activation = 0.0
    for i in range(len(row)):
        activation += weights[i] * row[i]
    return 1.0 if activation >= 0.0 else -1.0


# ____________________________________________________________________________________________________________________

# EVALUATING ALGORITHM

# ____________________________________________________________________________________________________________________

# method that splits dataset into n folds (subsets) for training and cross validation
def cross_validation_split(rows, n_folds):
    dataset_split = []
    dataset_copy = list(rows)
    fold_size = int(len(rows) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# method to evaluate predictions on my split dataset using n folds cross validation
def evaluate_algorithm(rows, n_folds, number_of_iterations, tolerance, learning_rate, boolean):
    # split into train and cv set
    folds = cross_validation_split(rows, n_folds)

    # initialize the lists
    list_of_train_precision = []
    list_of_train_recall = []
    list_of_train_accuracy = []

    list_of_test_precision = []
    list_of_test_recall = []
    list_of_test_accuracy = []

    index = 0
    i = 0

    # creating training and testing sets
    for fold in folds:
        # here is the training dataset
        train_set = list(folds)
        train_set = numpy.array(train_set)
        train_set = numpy.delete(train_set, (index), axis=0)
        index += 1
        train_set = train_set.tolist()
        train_set = sum(train_set, [])

        # create training labels
        y_train = []
        for i in range(len(train_set)):
            y_train.append(train_set[i][-1])
        for row in train_set:
            del row[-1]

        # get the mean and standard deviation on the train dataset
        mean, standard_deviation = mean_standard_deviation(train_set)
        # normalize the features in my training dataset
        z_score_normalize(train_set, mean, standard_deviation)

        # here is the testing dataset
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        for row in test_set:
            del row[-1]
        # normalize the features in my testing dataset
        z_score_normalize(test_set, mean, standard_deviation)

        # create target labels for test data
        y_test = [row[-1] for row in fold]

        # call perceptron_algorithm() to estimate weights and make predictions
        accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test = \
            perceptron_algorithm(train_set, test_set, y_train, y_test, number_of_iterations, tolerance, learning_rate,
                                 boolean)

        list_of_train_accuracy.append(accuracy_train)
        list_of_train_precision.append(precision_train)
        list_of_train_recall.append(recall_train)
        list_of_test_accuracy.append(accuracy_test)
        list_of_test_precision.append(precision_test)
        list_of_test_recall.append(recall_test)

    return list_of_train_precision, list_of_train_recall, list_of_train_accuracy, list_of_test_precision, \
           list_of_test_recall, list_of_test_accuracy


# ____________________________________________________________________________________________________________________

# PREDICTING PERCEPTRON

# ____________________________________________________________________________________________________________________

def main():
    seed(1)
    print("------->>>------>>>\n")
    print(f'PERCEPTRON RESULTS ')
    print("------->>>------>>>\n")

    filename_perceptron = load_csv('/Users/beccam/Downloads/perceptronData.csv')

    boolean = True

    # using these parameters for gradient_descent():
    tolerance = 0.001  # select the optimal values
    learning_rate = 1e-3

    # use ten-fold cross-validation to calculate precision, recall, accuracy for each fold and their means
    n_folds = 10

    # set a max number of iterations (recommended max iterations = 50000)
    number_of_iterations = 50000  # select the optimal values

    train_precision, train_recall, train_accuracy, test_precision, test_recall, test_accuracy = \
        evaluate_algorithm(filename_perceptron, n_folds, number_of_iterations, tolerance, learning_rate, boolean)
    boolean = False

    print('')
    # report the precision, recall, accuracy of ten fold cv for 3 datasets using logistic regression for each fold
    print("train precision ", end='')
    print(train_precision)
    print('')
    print("mean train precision is ", end='')
    print(statistics.mean(train_precision))
    print('')
    print("test precision ", end='')
    print(test_precision)
    print('')
    print("mean test precision is ", end='')
    print(statistics.mean(test_precision))
    print('\n')

    print("train recall ", end='')
    print(train_recall)
    print('')
    print("mean train recall is ", end='')
    print(statistics.mean(train_recall))
    print('')
    print("test recall ", end='')
    print(test_recall)
    print('')
    print("mean test recall is ", end='')
    print(statistics.mean(test_recall))
    print('\n')

    print("train accuracy ", end='')
    print(train_accuracy)
    print('')
    print("mean train accuracy is ", end='')
    print(statistics.mean(train_accuracy))
    print('')
    print("test accuracy ", end='')
    print(test_accuracy)
    print('')
    print("mean test accuracy is ", end='')
    print(statistics.mean(test_accuracy))
    print('\n')


if __name__ == "__main__":
    main()
