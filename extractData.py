import matplotlib.pyplot as plt
import numpy as np
import csv

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

with open('csv/Salary_predict.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    X = []
    y = []
    for row in readCSV:
        x = []
        Y = []
        for j in range (len(row)):
            if (isFloat(row[j])):
                if j != len(row) - 1:
                    x.append(float(row[j]))
                else:
                    Y.append(float(row[j]))
        
        if (len(x) != 0):
            X.append(x)
            y.append(Y)
    
    print(len(X), len(X[0]))
    for i in range (len(X)):
        for j in range(len(X[i])):
            print(X[i][j], end=" ")
        
        for j in range(len(y[i])):
            print(y[i][j])
