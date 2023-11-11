import fileinput
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import datasets, linear_model

from sklearn.preprocessing import MinMaxScaler

# Open data file
#sys.stdin = open('data/YearExperience_Salary_3.data', 'r')

# Read n, m respectively number of data and number of argument in single data.
[n, m] = map(int, input().split())

# Initialize matrix X and y
X = np.zeros((n, m))
y = np.zeros((n, 1))

numIterations = 1000

# Read X and y
for i in range(n):
	line = input().split()
	
	for j in range(m):
		X[i][j] = line[j]
	
	y[i] = line[m]


# function to show data point.
def showData(X, y, x_min, x_max, y_min, y_max):
	plt.plot(X, y, 'ro')
	plt.axis([x_min, x_max, y_min, y_max])
	plt.xlabel('Height (cm)')
	plt.ylabel('Weight (kg)')
	plt.show()

def showData(w, x_min, x_max, y_min, y_max):
	x0 = np.linspace(x_min, x_max, 1000)
	y0 = w[0][0] + w[1][0] * x0

	plt.plot(X, y, 'ro')
	plt.plot(x0, y0)
	plt.axis([x_min, x_max, y_min, y_max])
	plt.xlabel('Height (cm)')
	plt.ylabel('Weight (kg)')
	plt.show()

#showData(X, y, 100, 200, 0, 100)

def solveWithMatrix(X, y):
	ones = np.ones((X.shape[0], 1))
	Xbar = np.concatenate((ones, X), axis=1)

	# Calculate w
	A = np.dot(Xbar.T, Xbar)
	b = np.dot(Xbar.T, y)
	w = np.dot(np.linalg.pinv(A), b)

	print("Coefficient solution by formula: ")
	print(w)

	print("Cost by formula: ")
	print(cost(Xbar, y, w))
	print()

def cost(X, y, w):
	n = X.shape[0]
	return .5 /n * np.linalg.norm(np.dot(X, w) - y, 2) ** 2


def costTransformed(X, y, w, scalery: MinMaxScaler):
	n = X.shape[0]
	return .5 /n * np.linalg.norm(scalery.inverse_transform(np.dot(X, w)) - scalery.inverse_transform(y), 2) ** 2


def calculateGradient(X, y, w):
	eps = 1e-6
	g = np.zeros_like(w)

	for i in range(w.shape[0]):
		w_inc = w.copy()
		w_dec = w.copy()

		w_inc[i] += eps
		w_dec[i] -= eps

		g[i] = (cost(X, y, w_inc) - cost(X, y, w_dec)) / (2 * eps)
	
	return g

#calculate gradient
def grad(X, y, w):
	n = X.shape[0]
	return 1 /n * X.T.dot(np.dot(X, w) - y)

def gradientDescent(X, y, w_init, grad_func, eta):
	w_list = [w_init]

	for it in range(numIterations):
		# calculate w_new = w - eta * gradient
		w_new = w_list[-1] - grad_func(X, y, w_list[-1]) * eta

		if np.linalg.norm(grad_func(X, y, w_new))/len(w_new) < 1e-6:
			break

		#print(w_new)
		# add w_new to w_list
		w_list.append(w_new)

	return (w_list, it)

"""
def gradientDescentWithMomentum(X, y, w_init, grad_func, eta, gamma):
	w_list = [w_init]
	v_old = np.zeros_like(w_init)

	for it in range(numIterations):
		v_new = v_old * gamma + eta * grad_func(X, y, w_list[-1])
		w_new = w_list[-1] - v_new

		if (np.linalg.norm(grad_func(X, y, w_new))/len(w_new) < 1e-3):
			break

		w_list.append(w_new)
		v_old = v_new

	return (w_list, it)

def gradientDescentWithNAG(X, y, w_init, grad_func, eta, gamma):
	w_list = [w_init]
	v_old = np.zeros_like(w_init)

	for it in range(numIterations):
		v_new = v_old * gamma + eta * grad_func(X, y, w_list[-1] - gamma * v_old)
		w_new = w_list[-1] - v_new

		if (np.linalg.norm(grad_func(X, y, w_new))/len(w_new) < 1e-3):
			break

		w_list.append(w_new)
		v_old = v_new

	return (w_list, it)
"""
	
def solveByGradientDescent(X, y):
	scalerx = MinMaxScaler()
	scalery = MinMaxScaler()
	Xtrans = scalerx.fit_transform(X)
	ytrans = scalery.fit_transform(y)
	ones = np.ones((Xtrans.shape[0], 1))
	Xbar = np.concatenate((ones, Xtrans), axis=1)

	w_init = np.zeros((Xbar.shape[1], 1))
	# print(Xbar)

	[w_list, it] = gradientDescent(Xbar, ytrans, w_init, calculateGradient, 0.6)
	# print(scalerx.transform([[0]]))
	# print(scalerx.data_min_, scalerx.data_max_, scalerx.data_range_)
	# print(scalery.data_min_, scalery.data_max_, scalery.data_range_)
	# print(w_list[-1])
	temp = w_list[-1].copy()
	# # print(Xbar)
	# # print(scalerx.data_range_)
	# # print(scalery.data_range_)
	# zeros = np.zeros((scalerx.data_max_.shape[0], 1))
	# print(np.concatenate(([[1]], scalerx.transform(zeros.T)), axis=1))
	temp[0] = scalery.inverse_transform(np.concatenate(([[1]], scalerx.transform(np.zeros((scalerx.data_max_.shape[0], 1)).T)), axis=1).dot(w_list[-1]))[0]
	for i in range(scalerx.data_range_.shape[0]):
		temp[i + 1][0] = temp[i + 1][0] / scalerx.data_range_[i] * scalery.data_range_[0]
	
	print("Coefficient solution by gradient descent: ")
	print(temp)
	# print(scalery.inverse_transform(temp))
	# print("trans x ", scalerx.transform([[scalerx.data_max_[0]], [0]]))
	# print("trans x2", scalerx.transform([[scalerx.data_max_[0]], [1]]))
	# temp2 = scalerx.transform([[scalerx.data_max_[0]], [0]]).T.dot(temp)
	# temp3 = scalery.inverse_transform(scalerx.transform([[scalerx.data_max_[0]], [1]]).T.dot(temp)) - scalery.inverse_transform(temp2);
	# temp4 = scalerx.transform([[scalerx.data_max_[0]], [1]]).T.dot(temp) - temp2
	# print(temp2, scalery.inverse_transform(temp2))
	# print(temp3)
	# print(temp4)
	# print(scalerx.transform([[scalerx.data_max_[0]], [scalerx.data_min_[0]]]).T.dot(temp), scalery.inverse_transform(scalerx.transform([[1], [0]]).T.dot(temp)))
	# print(scalerx.transform([[scalerx.data_min_[0]], [scalerx.data_max_[0]]]).T.dot(temp), scalery.inverse_transform(scalerx.transform([[0], [1]]).T.dot(temp)))
	# print(costTransformed(Xbar, ytrans, w_list[-1], scalery))
	
	print("Cost by gradient descent: ")
	print(cost(np.concatenate((ones, X), axis=1), y, temp))

	print("Number of iterations: ")
	print("it = ", it)
	print()

	# tempx = np.array([[0, 200]]).T
	# tempxtrans = scalerx.transform(tempx)
	# #print(tempxtrans)
	# tempones = np.ones((tempx.shape[0], 1))
	# tempxbar = np.concatenate((tempones, tempxtrans), axis=1)
	# tempy = tempxbar.dot(w_list[-1])
	# #print(tempy)
	# tempy = scalery.inverse_transform(tempy)
	# print("y:", scalery.inverse_transform(tempxbar.dot(w_list[-1])))

	# plt.plot(X, y, 'ro')
	# plt.plot(tempx, tempy)
	# plt.axis([0, 200, -100, 100])
	# plt.xlabel('Height (cm)')
	# plt.ylabel('Weight (kg)')
	# plt.show()
	# print(tempy[0])
	# showData(w_list[-1], 100, 200, 0, 100)

"""
def solveByGradientDescentWithMomentum(X, y):
	ones = np.ones((X.shape[0], 1))
	Xbar = np.concatenate((ones, X), axis=1)

	w_init = np.zeros((Xbar.shape[1], 1))

	[w_list, it] = gradientDescentWithMomentum(Xbar, y, w_init, grad, 1e-4, 0.9)

	print(w_list[-1])
	print(cost(Xbar, y, w_list[-1]))
	print("it = ", it)

	showData(w_list[-1], 100, 200, 0, 100)

def solveByGradientDescentWithNAG(X, y):
	ones = np.ones((X.shape[0], 1))
	Xbar = np.concatenate((ones, X), axis=1)

	w_init = np.zeros((Xbar.shape[1], 1))
	w_init[0][0] = -20

	[w_list, it] = gradientDescentWithNAG(Xbar, y, w_init, grad, 1e-4 /2, 0.7)

	print(w_list[-1])
	print(cost(Xbar, y, w_list[-1]))
	print("it = ", it)

	showData(w_list[-1], 100, 200, 0, 100)
"""

def solveByLibrary(X, y):
	ones = np.ones((X.shape[0], 1))
	Xbar = np.concatenate((ones, X), axis=1)
	regr = linear_model.LinearRegression(fit_intercept=False)
	regr.fit(Xbar, y)

#   print(regr.score(Xbar, y))

	w = regr.coef_

	print("Coefficient solution by library: ")
	print(w.T)

	print("Cost by library: ")
	print(cost(Xbar, y, w.T))

	print()
	
	#showData(w.T, 100, 200, 0, 100)

# for i in range(n):
# 	for j in range(m):
# 		print(f"{X[i][j]:.2f}", end=" ")
# 	print(f"{y[i][0]:.2f}")
# scalerx = MinMaxScaler()
# scalery = MinMaxScaler()
# print()
# X = scalerx.fit_transform(X)
# y = scalery.fit_transform(y)
# for i in range(n):
# 	for j in range(m):
# 		print(f"{X[i][j]:.2f}", end=" ")
# 	print(f"{y[i][0]:.2f}")
solveWithMatrix(X, y)
solveByLibrary(X, y)
solveByGradientDescent(X, y)
#solveByGradientDescentWithMomentum(X, y)
#solveByGradientDescentWithNAG(X, y)

#main()