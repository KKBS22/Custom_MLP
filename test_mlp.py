import numpy as np
import pandas as pd
import pickle as pkl
import csv
from acc_calc import accuracy 


STUDENT_NAME = ['ARMINDER KAUR CHAHAL','KAUSHAL KIRAN BANGALORE SMAPATH KUMAR','YASHASWINI YOGESH'] 
STUDENT_ID = ['20918502','20877642','20926965']

def test_mlp(data_file):
    # Load the testpkl set
    # START
    Data = pd.read_csv(data_file).values
    #rain_labels = pd.read_csv('D:\\UniversityOfWaterloo\\Courses\\657_ToolsOfIntelligentSystemDesign\\Assignment1\\train_label_orig.csv').values
    # END
    # Load your network
    # START
    wOne =0

    with open('weightsOneV1.pkl', 'rb') as f:
        wOne = pkl.load(f)
    with open('weightsTwoV2.pkl', 'rb') as f1:
        wTwo = pkl.load(f1)
    with open('biasOne.pkl', 'rb') as f2:
        bOne = pkl.load(f2)
    with open('biasTwo.pkl', 'rb') as f3:
        bTwo = pkl.load(f3)

    #print(type(wOne))
    #print(type(wTwo))
    #print(type(bOne))
    #print(type(bTwo))

    result_Hotcoded = classifier(Data,wOne, wTwo, bOne, bTwo)

    return result_Hotcoded

    #np.savetxt("foo.csv", result_Hotcoded, delimiter=",")


def soft_max(x):
    exponent = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exponent/np.sum(exponent, axis=1, keepdims=True)

def sigmoid(x):
    output = 1 / (1 + np.exp(-x) )
    return output
    
#def classifier(Data, wOne, wTwo, bOne, bTwo, rain_labels):

def classifier(Data, wOne, wTwo, bOne, bTwo):
    output_layer1 = np.dot(Data, wOne) + bOne
    output_layer1 = sigmoid(output_layer1)
    
    output_layer2 = np.dot(output_layer1, wTwo) + bTwo
    output_layer2 = soft_max(output_layer2)
    #print(output_layer2.shape[0])
    #print(output_layer2[0])
    y_pred = np.zeros([Data.shape[0],4])
    
    for a in range(output_layer2.shape[0]):
        y_pred[a][np.where(output_layer2[a] == np.max(output_layer2[a]))] = 1
    #test_accuracy = accuracy(rain_labels, y_pred)*100
    #print(test_accuracy)
    return y_pred

#def main():
    test_mlp('D:\\UniversityOfWaterloo\\Courses\\657_ToolsOfIntelligentSystemDesign\\Assignment1\\train_data_orig.csv')


#if __name__ == '__main__':
   # main()
'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''