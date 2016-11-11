#!/usr/bin/env python

import numpy as np
from math import log
import operator
from task_1 import *

'''
This program is using the decision tree learning algorithm to train the machine
to classify the good and bad data, in order to classify any new coming data.
However, due to time limitation, we only can print the decision tree from the
training data in this assignment.

Author: Kwok Shing Lam (Student No. n9516778)
Contributor: Jack Lee (Student No. n7105819)
Date: 6 June 2016
'''

# Declare constraints
POSITIVE_CLASSIFICATION = 1
GOAL = "A16"

'''
This function calculate the probability of a value k appears in the random variable.
Pre-condition: vk not null,
               var_set has at least one element
Post-condition: return the probability of the value
vk : the k-th value of a random variable
var_set : an array that stored all the elements in a random variable
'''
def calProbabilityOfValueInRandomVariable(vk, var_set):
    size_of_subset_vk = 0
    if type(var_set) == np.ndarray:
        # if the input var_set is more than one element, then calculate how
        # many element in the array that match vk
        
        size_of_vset = len(var_set)

        # accumulalte the size of subset vk
        for elem in var_set:
            if (elem == vk):
                size_of_subset_vk += 1
    else:
        # if the input var_set has only one element, than the probability
        # of vk should be equal to 1
        size_of_vset = 1
        if (var_set == vk):
            size_of_subset_vk += 1
        
    return float(size_of_subset_vk) / size_of_vset

'''
This function calculate the entropy of a given value from the random variable.
Pre-condition: var_value not null
               var_set has a least one element
Post-condition: return the entropy of the given
var_values : an array stored all the values in the random
var_set : an array that stored all the elements in a random variable
'''
def calEntropy(var_values, var_set):
    # Entropy formula is obtained from
    # Russell, S. & Norvig, P. (2010). Artificial Intelligence
    # - A Modern Approach (3rd Ed.), p.704. Upper Saddle River,
    # NJ: Pearson Education.
    entropy = 0.0
    for vk in var_values:
        p_vk = calProbabilityOfValueInRandomVariable(vk, var_set)
        if not (p_vk == 0):
            entropy += (p_vk * log(p_vk,2)) 
    return entropy * -1 

'''
This function calculate the total number of positive examples in a dataset
Pre-condition: goal not null
               dataset not null
Post-condition: the number of positive examples in this dataset
goal : the variable name classifying the dataset
dataset: an array that stored all the training data
'''
def calNumPositiveExamples(goal, dataset):
    goal_state = POSITIVE_CLASSIFICATION
    num = 0
    if type(dataset[goal]) == np.ndarray:
        # do iteration if the input dataset has more than one value
        for elem in dataset[goal]:
            if elem == goal_state:
                num += 1
    else:
        # check the only value from the input dataset
        if dataset[goal] == goal_state:
            num += 1
    return num

'''
This function calculate the total number of negative examples in a dataset
Pre-condition: goal not null
               dataset not null
Post-condition: the number of negative examples in this dataset
goal : the variable name classifying the dataset
dataset: an array that stored all the training data
'''
def calNumNegativeExamples(goal, dataset):
    goal_state = POSITIVE_CLASSIFICATION
    num = 0
    if type(dataset[goal]) == np.ndarray:
        # do iteration if the input dataset has more than one value
        for elem in dataset[goal]:
            if not elem == goal_state:
                num += 1
    else:
        # check the only value from the input dataset
        if not dataset[goal] == goal_state:
            num += 1
    return num

'''
This function slices the given dataset into subset of vk
Pre-condition: dataset not null
               attr not null
               var_values not null
Post-condition: the number of sliced subset is equal to the number of variable
                values
dataset : an array that stored all the training data
attr : the current attribute we are looking for slicing
var_values : an array that stored the distinct values in the random variable
'''
def sliceSubset(dataset, attr, var_values):
    subsets = {}
    for record in dataset:
        for vk in var_values:
            if record[attr] == vk:
                if vk not in subsets.keys():
                    # create a new subset of vk in the dictionary
                    # if it is not exist
                    subsets[vk] = record
                else:
                    # append to the existing subset
                    subsets[vk] = np.append(subsets[vk], record)
    return subsets

'''
This function calculate the remainder of the other attributes apart from the
current attribute. the remainder value is useful in calculate the information
gain.
Pre-condition: dataset not null
               attr not null
               var_values not null
               goal_var not null
Post-condition: the remainder is returned
dataset : an array that stored all the training data
attr : the attribute we are looking for slicing into subsets
var_values : an array that stored the distinct values in the random variable
goal_var : the attribute label that specify the goal classification

'''
def calRemainder(dataset, attr, var_values, goal_var):
    remainder = 0.0
    size_of_positive_examples_in_total = calNumPositiveExamples(goal_var, dataset)
    size_of_negative_examples_in_total = calNumNegativeExamples(goal_var, dataset)

    # slice the entire set into subsets
    subsets = sliceSubset(dataset, attr, var_values)

    # perform sumation to get the remainder:
    for vk in var_values:
        size_of_positive_examples_in_vk = calNumPositiveExamples(goal_var, subsets[vk])
        size_of_negative_examples_in_vk = calNumNegativeExamples(goal_var, subsets[vk])
        
        # Remainder formula (Russell & Norvig, 2010, p.704)
        remainder += float(size_of_positive_examples_in_vk + size_of_negative_examples_in_vk)\
                     / (size_of_positive_examples_in_total + size_of_negative_examples_in_total)\
                     * calEntropy(var_values, subsets[vk][attr])
        
    return remainder

'''
This function calculate the information gain of a specified attribute
Pre-condition: dataset not null
               attr not null
               var_values not null
               goal_var not null
Post-condition: the value of information gain is returned
dataset : an array that stored all the training data
attr : the attribute we are looking for slicing into subsets
var_values : an array that stored the distinct values in the random variable
goal_var : the attribute label that specify the goal classification
'''
def calInformationGain(dataset, attr, var_values, goal_var):
    return calEntropy(var_values, dataset[attr]) - calRemainder(dataset, attr, var_values, goal_var)

'''
This function extracts the distinct possible values from the random variable.
Pre-condition: dataset not null
               No. of labels > 0
Post-condition: a set of distinct values is returned
daaset : an array that stored all the training data
labels : the name set of all attributes
'''
def extractDistinctRandomVariableValues(dataset, labels):
    var_set = {}
    for variable in labels:
        var_set[variable] = []

        # in this for-loop, we will ignore the last attribute which is the class of the sample
        for elem in dataset[variable][:-1]:
            if elem not in var_set[variable]:
                # only the element not seen is append to the set
                var_set[variable].append(elem)
                
    return var_set

'''
This function finds the most important attribute from a set of attributes
Pre-condition: dataset not null
               No. of labels > 0
               var_set not null
               goal_label not null
Post-condition: the most important attribute's label in returned
daaset : an array that stored all the training data
labels : the name set of all attributes
var_set : the random variable set that stored the distinct values of each variable
goal_label : the attribute label that specify the goal classification 
'''
def findImportantAttr(dataset, labels, var_set, goal_label):
    max_gain = 0;
    important_key = labels[0]
    for variable in labels:
        gain = calInformationGain(dataset, variable, var_set[variable], goal_label)
        if gain > max_gain:
            max_gain = gain
            important_key = variable
    return important_key

'''
This function find the plurality value of a list of samples, which is the most occurance
of a classification
Pre-condition: examples > 0
Post-condition: a class is returned
examples : the curren example list for inspection
'''
def pluralityValue(examples):
    size_of_positive_examples = calNumPositiveExamples(GOAL, examples)
    size_of_negative_examples = calNumNegativeExamples(GOAL, examples)

    if (size_of_positive_examples > size_of_negative_examples):
        # classify these example as positive
        return 1
    elif (size_of_negative_examples > size_of_positive_examples):
        # classify these example as negative
        return -1
    else:
        # default is -1 if we cannot classify because the probability of positive
        # and negative classification is equal
        return -1

'''
This function determine if the given exampels are all belong the same class.
Pre-condition: examples not null
               goal not null
Post-condition: return true if all data is either in positive or in negative class
                return false if not all data belongs to the same class
examples : the example list for inspection
goal : the attribute label that relevant to the goal classification
'''
def haveSameClassfication(examples, goal):
    positive = 0
    negative = 0
    
    for elem in examples:
        if elem[goal] == 1:
            positive += 1
        elif elem[goal] == -1:
            negative += 1

    return (positive == len(examples) or negative == len(examples))

'''
This function is recursively building the tree from the most information gain attribute
down to the leaf until the attribute can classify the examples.
Pre-condition: No. of examples > 0
               attributes no null
               parent_examples > 0
               depth is positive integer
Post-condition: the structure of the decision tree is returned in formatted string
examples : the current example list for the learning process
attributes : the list of attributes for examples
parent_examples: the root of current node in the current subtree in case this node has no data
depth : the depth of the current node in the complete tree, used for indentation of the result
'''
def decisionTreeLearning(examples, attributes, parent_examples, depth=1):
    if examples.size == 0:
        return pluralityValue(parent_examples)
    elif examples.size == 1:
        return examples[GOAL]
    elif haveSameClassfication(examples, GOAL):
        return examples[0][GOAL]
    elif not attributes:
        return pluralityValue(examples)
    else:
        # find out the distinct values in the examples
        var_set = extractDistinctRandomVariableValues(examples, labels)
        
        #pop the new tree node from the labels list
        node_key = findImportantAttr(examples, labels, var_set, GOAL)
        labels.remove(node_key)

        # slice the current examples into subset of vk in this node attribute
        subsets = sliceSubset(examples, node_key, var_set[node_key])
        
        tree_path = ""
        tree_path += "" + node_key + "(\n"

        # Try the expand the sub-tree because we cannot classify in the current node
        for vk in var_set[node_key]:
            subtree = decisionTreeLearning(subsets[vk], labels, examples, depth + 1)
            for i in range(depth):
                tree_path += '\t'
            tree_path += "%s" % vk + ":%s\n" % subtree
            
        for i in range(depth-1):
            tree_path += '\t'
        tree_path += ")"
        for i in range(depth-1):
            tree_path += '\t'
            
        return tree_path


### MAIN BODY ###
data_source = task_1()

train_data = data_source.getTrainData()
n_train_data = data_source.getTrainDataSize()
labels = data_source.getLabels()
attrTypes = data_source.getAttrType()

print decisionTreeLearning(train_data, labels, train_data)
print "Negative attribute indicate it is a missing data."
