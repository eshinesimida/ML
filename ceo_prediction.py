# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:08:53 2019

@author: lenovo
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from computeCost import *
from gradientDescent import *
from plotData import *

def plot_data(x, y):
    # ===================== Your Code Here =====================
    # Instructions : Plot the training data into a figure using the matplotlib.pyplot
    #                using the "plt.scatter" function. Set the axis labels using
    #                "plt.xlabel" and "plt.ylabel". Assume the population and revenue data
    #                have been passed in as the x and y.

    # Hint : You can use the 'marker' parameter in the "plt.scatter" function to change the marker type (e.g. "x", "o").
    #        Furthermore, you can change the color of markers with 'c' parameter.


    # ===========================================================
    plt.scatter(x,y)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    

    # ==========================================================
    
    cost = sum((y - np.dot(X,theta))**2)/(2*m)
    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)
    lr_theta = np.zeros(2)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # Hint: X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )


        # ===========================================================
        # Save the cost every iteration
        theta1 = np.zeros(2)
       
        for n in range(m):
            theta1 = theta1 - 2.0*(y[n] - np.dot(X[n],theta))*X[n]
        
        #lr_b = lr_b + b_grad **2
        #lr_w = lr_w + w_grad**2
        lr_theta = lr_theta + theta1**2
        
        
        #b = b - lr/np.sqrt(lr_b) * b_grad
        #w = w - lr/np.sqrt(lr_w) * w_grad
        theta = theta - alpha/np.sqrt(lr_theta) * theta1
        
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)
    lr_theta = np.zeros(2)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #


        # ===========================================================
        # Save the cost every iteration
        theta1 = np.zeros(2)
       
        for n in m:
            theta1 = theta1 - 2.0*(y[n] - np.dot(X[n],theta))*X[n]
        
        #lr_b = lr_b + b_grad **2
        #lr_w = lr_w + w_grad**2
        lr_theta = lr_theta + theta1**2
        
        
        #b = b - lr/np.sqrt(lr_b) * b_grad
        #w = w - lr/np.sqrt(lr_w) * w_grad
        theta = theta - alpha/np.sqrt(lr_theta) * theta1
        
        
        #stroe parametrs for plotting
        #b_history.append(b)
        #w_history.append(w)
        
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
#================Part 1: Plotting ==============
print('Plotting Data ....')
data = np.loadtxt('street&profits.txt',delimiter = ',')
X = data[:,0]
y = data[:,1]
m = y.size
plot_data(X,y)

#============part 2:  Gradient descent
print('Running Gradient Descent')

X = np.c_[np.ones(m),X]#add a column of ones to X
theta = np.zeros(2)# initialize fitting parameters

#some gradient descent settings
iterations = 1500
alpha = 0.01

compute_cost(X, y, theta)

theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# Plot the linear fit
plt.figure(0)
line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
plt.legend(handles=[line1])
plot_data(data[:,0], data[:,1])

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))


# ===================== Part 3: Visualizing J(theta0, theta1) =====================
print('Visualizing J(theta0, theta1) ...')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

xs, ys = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros(xs.shape)

# Fill out J_vals
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t)

J_vals = np.transpose(J_vals)

fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()

plt.figure(2)
lvls = np.logspace(-2, 3, 20)
plt.contour(xs, ys, J_vals, levels=lvls, norm=LogNorm())
plt.plot(theta[0], theta[1], c='r', marker="x")
plt.show()
