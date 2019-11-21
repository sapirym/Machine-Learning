import sys
import numpy

class Algo:
    """
     Generic algorithm class.
    """
    def predict(self, weights, testData):
        """
        This function predicts the label over a test set and a weight matrix
        :param weights: initial weights.
        :param testData: test file.
        """
        predictions = []
        # go over the test set, predict and save the prediction
        for index in range(len(testData)):
            x = testData[index]
            prediction = numpy.argmax(numpy.dot(weights, x))
            predictions.append(prediction)
        return predictions

    def runAlgo(self, train_x, train_y, testData):
        """
        Run the algorithm.
        :param train_x: train values.
        :param train_y: trains labels
        :param testData:
        :return: The predictions of the algorithm.
        """
        w = self.train(train_x, train_y)
        predictions = self.predict(w, testData)
        return predictions


class Perceptron(Algo):
    """
    Perceptron class - inherits from Algo class.
    """

    def train(self,train_x, train_y):
        """
        Train the model.
        :param train_x: train values.
        :param train_y: trains labels
        :return: the final weights
        """
        length = len(train_x[0])
        possibleLabelsVals = set(train_y)

        # learning parameters
        w = numpy.zeros((len(possibleLabelsVals), length))
        epochs = 10
        examplesLen = len(train_x)
        eta = 0.1

        # train model
        for e in range(epochs):
            for index in range(examplesLen):
                x = train_x[index]
                y = train_y[index]
                y_hat = numpy.argmax(numpy.dot(w, x))
                if y != y_hat:
                    # update weights
                    w[int(y), :] = w[int(y), :] + eta * x
                    w[int(y_hat), :] = w[int(y_hat), :] - eta * x
            eta /= e + 1
        return w


class PA(Algo):
    """
        PA class - inherits from Algo class.
    """

    def train(self, train_x, train_y):
        """
        Train the model.
        :param train_x: train values.
        :param train_y: trains labels
        :return: the final weights
        """
        length = len(train_x[0])
        possibleLabelsVals = set(train_y)
        w = numpy.zeros((len(possibleLabelsVals), length))

        # learning parameters
        epochs = 10
        examplesLen = len(train_x)

        # train model
        for e in range(epochs):
            for index in range(examplesLen):
                x = train_x[index]  # data
                y = int(train_y[index])  # label
                y_hat = int(numpy.argmax([numpy.dot(x, w_i) for w_i in w]))
                tau = self.calculateTau(x,y,w,y_hat)
                # update
                if y != y_hat:
                    w[int(y), :] = w[int(y), :] + tau * x
                    w[int(y_hat), :] = w[int(y_hat), :] - tau * x
        return w


    def calculateTau(self,x,y,w,y_hat):
        """
        Calculate tau
        :param x: the values
        :param y: actual label
        :param w: weights
        :param y_hat: predicted label
        :return: updated tau
        """
        l_value = max(0,1 - numpy.dot(w[y],x) + numpy.dot(w[y_hat],x))
        norm_x = (numpy.linalg.norm(x))**2
        if norm_x == 0:
            tau = 0
        else:
            tau = l_value/(2*norm_x)
        return tau


class SVM(Algo):
    """
        PA class - inherits from Algo class.
    """

    def train(self, train_x, train_y):
        """
        Train the model.
        :param train_x: train values.
        :param train_y: trains labels
        :return: the final weights
        """
        length = len(train_x[0])
        possibleLabelsVals = set(train_y)
        w = numpy.zeros((len(possibleLabelsVals), length))

        # learning parameters
        epochs = 10
        examplesLen = len(train_x)
        eta = 0.015
        lamda = 0.2

        # training
        for e in range(epochs):
            for index in range(examplesLen):
                x = train_x[index]  # data
                y = train_y[index]  # label
                y_hat = numpy.argmax(numpy.dot(w, x))

                if y != y_hat:
                    # update weights
                    w[int(y), :] = (1-eta*lamda)*w[int(y), :] + eta * x
                    w[int(y_hat), :] = (1-eta*lamda)*w[int(y_hat), :] - eta * x
                    for index in range(len(possibleLabelsVals)):
                        if index != y and index != y_hat:
                            w[index, :] = (1 - eta * lamda) * w[int(index), :]
        return w



def convertSex(sex_name):
    """
    Converts sex name to array of ints.
    :param sex_name: all options of sex type
    :return: array of ints.
    """
    returnValue = [0,0,0]
    if(sex_name == "M"):
        returnValue[0] = 1
    elif(sex_name == "F"):
        returnValue[1] = 1
    else:
        returnValue[2] = 1
    return  returnValue

"""
# Calculate accuracy precentage.

def calculate(arr, train_y_file):
    counter = 0
    totalElements = len(arr)
    with open(train_y_file) as fp:
        line = fp.readline().replace("\n","")
        line_index = 0
        while line:
            if arr[line_index] == float(line):
                counter += 1
            line = fp.readline().replace("\n", "")
            line_index += 1
    print(counter/totalElements)
    
def calculate2(arr, test_y_list):
    counter = 0
    totalElements = len(arr)
    for i,line in enumerate(test_y_list):
        if arr[i] == line:
            counter += 1
    print(counter/totalElements)
"""

def readTrainData(train_x_file):
    """
    Reads the training data and normalize it.
    :param train_x_file: training values.
    :return: normalized train values, minVal, maxVal.
    """
    train_x = []
    minVal = []
    maxVal = []

    # open training values file
    with open(train_x_file) as fp:
        line = fp.readline().replace("\n", "")
        line_index = 0
        while line:
            line_vals = line.split(",")
            newLine = convertSex(line_vals[0])
            newLine.extend(line_vals[1:])
            train_x.append(numpy.array(newLine).astype(numpy.float))
            line = fp.readline().replace("\n", "")
            line_index += 1

    # normalize values
    train_x = numpy.array(train_x)
    train_x = numpy.transpose(train_x)
    for i, line in enumerate(train_x):
        minVal.append(line.min())
        maxVal.append(line.max())
        if line.max() != line.min():
            train_x[i] = (line - line.min()) / (line.max() - line.min())
    train_x = numpy.transpose(train_x)

    return train_x,minVal,maxVal


def readTestDate(testData, minVal, maxVal):
    """
    Reads the test data and normalize it.
    :param train_x_file: training values.
    :return: normalized train values, minVal, maxVal.
    """
    test_x = []
    with open(testData) as fp:
        line = fp.readline().replace("\n", "")
        line_index = 0
        while line:
            line_vals = line.split(",")
            newLine = convertSex(line_vals[0])
            newLine.extend(line_vals[1:])
            test_x.append(numpy.array(newLine).astype(numpy.float))
            line = fp.readline().replace("\n", "")
            line_index += 1
        test_x = numpy.array(test_x)
        test_x = numpy.transpose(test_x)

    # normalize the training set by MinMax.
    for i, line in enumerate(test_x):
        if minVal[i] != maxVal[i]:
            test_x[i] = (line - minVal[i]) / (maxVal[i] - minVal[i])
    test_x = numpy.transpose(test_x)

    return test_x

def readYData(train_y_file):
    """
    Reads the training labels and returns an array of the labels
    :param train_y_file: train labels
    :return: label file after parsing
    """
    train_y = []
    with open(train_y_file) as fp:
        line = fp.readline().replace("\n", "")
        line_index = 0
        while line:
            train_y.append(float(line))
            line = fp.readline().replace("\n", "")
            line_index += 1
    return train_y


if __name__ == '__main__':
    """
    main function of the app which runs three algorithms over the data
    """
    arguments = sys.argv[1:]
    train_x_file = "train_x.txt"#arguments[0] # Train
    train_y_file = "train_y.txt"#arguments[1] # Labels
    test_x_file = "test_x.txt"#arguments[2]  # Test

    # get the training data, label and test data
    train_x,minVal,maxVal = readTrainData(train_x_file)
    train_y = readYData(train_y_file)
    test_x = readTestDate(test_x_file,minVal,maxVal)

    # Run Algorithms

    # Perceptron
    perceptron = Perceptron()
    predictionsPerceptron = perceptron.runAlgo(train_x,train_y,test_x)
    #calculate(predictionsPerceptron,"test_y.txt")

    # SVM
    svm = SVM()
    predictionsSVM = svm.runAlgo(train_x,train_y,test_x)
    #calculate(predictionsSVM,"test_y.txt")

    # PA
    pa = PA()
    predictionsPa = pa.runAlgo(train_x, train_y, test_x)
    #calculate(predictionsPa, "test_y.txt")

    for a,b,c in zip(predictionsPerceptron,predictionsSVM,predictionsPa):
        print("perceptron: {}, svm: {}, pa: {}".format(a,b,c))
