"""
Text classification
"""

import util
import operator
from collections import Counter
import re
import numpy as np


class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 3 lines of code expected)
        if k == -1 or k > len(blacklist):
            self.blacklist = set(blacklist)
        else:
            self.blacklist = set(blacklist[0:k])
        # self.blacklist = set(blacklist)
        self.n = n
        self.k = k
        # raise NotImplementedError("TODO:")
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        badCnt = 0
        wordsInText = text.split()
        for word in wordsInText:
            if word in self.blacklist:
                badCnt += 1
            if badCnt >= self.n:
                return -1
        return 1
        # raise NotImplementedError("TODO:")
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$. 
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    # Create feature vector
    features = Counter()    # data structure similar to the one provided on the description page
                            # { brown: 1, lazy: 1, fence: 1, fox: 1, over: 1, chased: 1...}
    words = x.split()
    for word in words:
        features[word] += 1
    return features
    # raise NotImplementedError("TODO:")
    # END_YOUR_CODE


class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = params

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        features = self.featureFunction(x)
        weighted_sum = 0
        for feature in features.keys():
            if feature in self.params:
                weighted_sum += self.params[feature] * features[feature]
        return weighted_sum
        # raise NotImplementedError("TODO:")
        # END_YOUR_CODE

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('pos', 'neg'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    # For each iteration
    weights = dict()
    for i in range(iters):
        # For each training example
        for trainExample in trainExamples:
            # predict h
            features = featureExtractor(trainExample[0])
            label_i = trainExample[1]
            weights_sum = 0
            for feature in features:
                if feature in weights:
                    weights_sum += weights[feature] * features[feature]
            if weights_sum >= 0:
                if label_i == labels[1]: # ham
                    for feature in features:
                        if feature in weights.keys():
                            weights[feature] -= features[feature]
                        else:
                            weights[feature] = -features[feature]
            elif weights_sum < 0:
                if label_i == labels[0]: # spam
                    for feature in features:
                        if feature in weights.keys():
                            weights[feature] += features[feature]
                        else:
                            weights[feature] = features[feature]
    return weights

    # raise NotImplementedError("TODO:")
    # END_YOUR_CODE

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$. 

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    features = dict()    # data structure similar to the one provided on the description page
                            # { brown: 1, lazy: 1, fence: 1, fox: 1, over: 1, chased: 1...}
    # print(x)
    words = x.split()
    features['-BEGIN-' + " "+words[0]]=1
    bigram=[]
    for i in range(0, len(words)-1):
        bigram.append(words[i] +" "+ words[i+1])

    for word in words:
        if word in features.keys():
            features[word]+=1
        else:
            features[word] = 1

    for bi in bigram:
        if bi in features.keys():
            features[bi] += 1
        else:
            features[bi] = 1

    return features
    # raise NotImplementedError("TODO:")
    # END_YOUR_CODE

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.classifiers = dict()
        for classifier in classifiers:
            self.classifiers[classifier[0]] = classifier[1]
        # raise NotImplementedError("TODO:")
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        # self.classify(x) => returns [(str, double), .... , (str, double)]
        (maxLabel, maxScore) = max(self.classify(x), key=lambda k: k[1])
        return maxLabel
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        label_score = []
        for label in self.classifiers.keys():
            score = self.classifiers[label].classify(x)
            label_score.append((label, score))
        return label_score
        # raise NotImplementedError("TODO:")
        # END_YOUR_CODE

def learnOneVsAllClassifiers( trainExamples, featureFunction, labels, perClassifierIters = 10 ):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    # labels => ('comp', 'rec', 'talk', 'god', 'sci')
    # For each label => transform it into 'pos' or 'neg'
    binary_labels = ('pos', 'neg')
    ret = []
    for label_i in labels:
        trainExamplesMod = []
        for text, label in trainExamples:
            if label == label_i:        # if label is 'comp' -> 'pos' else 'neg'
                trainExamplesMod.append((text, 'pos'))
            else:
                trainExamplesMod.append((text, 'neg'))
        ret.append((label_i, WeightedClassifier(binary_labels, featureFunction, learnWeightsFromPerceptron(trainExamplesMod, featureFunction, binary_labels, perClassifierIters))))
    return ret

    # raise NotImplementedError("TODO:")
    # END_YOUR_CODE

