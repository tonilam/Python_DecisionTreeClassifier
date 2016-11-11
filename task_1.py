#!/usr/bin/env python

import numpy as np
import math
    
'''
This class is used to read a text file and randomly store the records into two seperate arrays:
1. training data
2. testing data

Author: Kwok Shing Lam (Student No. n9516778)
Contributor: Jack Lee (Student No. n7105819)
Date: 6 June 2016
'''
class task_1():
    
    ## Define constraints
    
    DEFAULT_FILE = "records.txt"
    TRAIN_PERCENTAGE = 0.8
    TEST_PERCENTAGE = 0.2
    
    # For the number listed here,
    # please refer to assignment specification page 2
    ATTRIBUTE_TYPE = np.array([2, -1, -1, 4, 3, 14, 9, -1,
                               2, 2, -1, 2, 3, -1, -1, 2])
    FIELD_LABELS = ('A1','A2','A3','A4','A5','A6','A7','A8',
                    'A9','A10','A11','A12','A13','A14','A15','A16')

    # The final version of the dataset should be all in numeric numbers.
    FIELD_DTYPE = ('int','int','int','int','int','int','int','int',
                   'int','int','int','int','int','int','int','int')

    ## END Define constraints

    ## Method definitions
    
    '''
    This is the constructor to load and generate arrays for further use
    '''
    def __init__(self, datafile=DEFAULT_FILE):

        # Generate a numpy array from the file
        data = self.readDataFromFile(datafile)

        # Shuffle the row
        np.random.shuffle(data)

        # determine how much data in a specific dataset
        n_data = data.shape[0]
        self.n_train = math.floor(n_data * self.TRAIN_PERCENTAGE)
        self.n_test = math.ceil(n_data * self.TEST_PERCENTAGE)

        # copy the data into two dataset
        self.train_data = data[:self.n_train]
        self.test_data = data[n_data-self.n_test:]

    '''
    This method read the data from a text file and convert all the data into numeric
    representation for further use
    Pre-condition: file exists
    Post-condition: a numpy array that stored the numeric set of data
    '''
    def readDataFromFile(self, file):
        # Define a lambda function to find the value that accurate to the integer place
        truncate1 = lambda x: int(float(x))
        
        # Define a lambda function to find the value that accurate to the tenths place
        truncate10 = lambda x: math.ceil(float(x) /10) * 10
        
        # Define a lambda function to find the value that accurate to the hundred place
        truncate100 = lambda x: math.ceil(float(x) /100) * 100
        
        # Define a lambda function to find the numeric representation of a character
        # Here we use the ascii value of the character - 96, so that 'a' = 1,
        # 'b' = 2, etc. Note: the missing data is -33 from this conversion.
        convertchar = lambda x: ord(x) - 96
        
        # Define a lambda function to find the numeric representation of a string
        # Here we use sum of the ascii values of each character, and then - 96,
        # so that 'a' = 1, 'b' = 2, 'aa' = 98, etc
        convertstring = lambda x: sum(map(ord,x)) - 96

        # Define a lambda function to find the numeric representation of '+' & '-'
        # Here we use 1 for '+' and -1 for '-'
        convertboolean = lambda x: (ord(x) - 44) * -1
        
        return np.genfromtxt(file,
                   delimiter=',',
                   dtype={'names': self.FIELD_LABELS,
                          'formats': self.FIELD_DTYPE},
                   converters={0:convertchar,
                               1: truncate10,
                               2: truncate10,
                               3:convertchar,
                               4:convertstring,
                               5:convertstring,
                               6:convertstring,
                               7: truncate1,
                               8:convertchar,
                               9:convertchar,
                               10: truncate1,
                               11:convertchar,
                               12:convertchar,
                               13: truncate100,
                               14: truncate100,
                               15:convertboolean},
                   missing_values=-1)
        
    '''
    This method returns the training data array
    Pre-condition: self.train_data not null
    Post-condition: self.train_data is returned
    '''
    def getTrainData(self):
        return self.train_data

    '''
    This method returns the testing data array
    Pre-condition: self.test_data not null
    Post-condition: self.test_data is returned
    '''
    def getTestData(self):
        return self.test_data

    '''
    This method returns the training data array size
    Pre-condition: self.n_train not null
    Post-condition: self.n_train is returned
    '''
    def getTrainDataSize(self):
        return self.n_train

    '''
    This method returns the testing data array size
    Pre-condition: self.n_test not null
    Post-condition: self.n_test is returned
    '''
    def getTestDataSize(self):
        return self.n_test

    '''
    This method returns the list of label's names
    Pre-condition: self.FIELD_LABELS not null
    Post-condition: the information stored in FIELD_LABELS is returned
    '''
    def getLabels(self):
        return list(self.FIELD_LABELS)

    '''
    This method returns the number of attribute types as a list
    Pre-condition: self.ATTRIBUTE_TYPE not null
    Post-condition: the information stored in ATTRIBUTE_TYPE is returned
    '''
    def getAttrType(self):
        return list(self.ATTRIBUTE_TYPE)

    ## END Method definitions

'''
This main method is used only for debug and testing purpose.
'''
if __name__ == "__main__":
    a = task_1()
    print "Training Data:"
    b = a.getTrainData()
    print b
