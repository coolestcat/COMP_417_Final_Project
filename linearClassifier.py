import random
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib

def getLine(params):
	linex = np.arange(-2,4,0.01)

	y = []
	for x in linex:
		y.append(params[0] + params[1]*x)

	return(linex,y)

def linearProbability(thisClass, params, point):
	m1 = params[1]
	b1 = params[0]

	if (m1!=0):
		m2 = -1/float(m1)
	else:
		m2 = 999999

	# b = y - mx
	b2 = point[1] - m2*point[0]

	# intersection x = (b2 - b1) / (m1-m2)
	x = (b2-b1)/float(m1-m2)
	y = m1*x + b1

	# print y
	y = m2*x + b2

	d = math.sqrt(math.pow((x-point[0]),2) + math.pow((y - point[1]),2))
	# print d
	k = 3
	# return d/float(k) # approximation
	return 1/float((math.exp(-1*k*d) + 1)) 

def getPosteriorProbs(params, datapoints):
	probs = []
	for point in datapoints:
		if point[1] > params[0] + (params[1]*point[0]):
			probs.append(linearProbability(0, params, point))
		else: 
			probs.append(1- linearProbability(0, params, point))

	return probs


# def costFunction(classifier, datapoints, params, nClasses):

# 	sum = 0
# 	for i in range(0, nClasses):
# 		withinVariance = 0
# 		for j in range(0, len(datapoints)-1):
# 			prob2 = classifier(i, params, datapoints[j+1])
# 			prob1 = classifier(i, params, datapoints[j])
# 			difference = prob2 - prob1

# 			differenceSquared = math.pow(difference,2)
# 			withinVariance += differenceSquared

# 		probs = []
# 		for point in datapoints:
# 			probs.append(classifier(i, params, point))
# 		var =  math.pow(np.std(probs),2)
# 		betweenVariance = math.pow(var, 2)

# 		sum += (withinVariance/float(betweenVariance))

# 	return sum

def costFunctionLinear(classifier, datapoints, params, nClasses):

	sum = 0
	withinVariance = 0
	for j in range(0, len(datapoints)-1):

		if datapoints[j+1][1] < params[0] + (params[1]*datapoints[j+1][0]):
			prob2 = classifier(i, params, datapoints[j+1])
			# print "first class: " + str(prob2)
		else:
			prob2 = 1- classifier(i, params, datapoints[j+1])
			# print "second class: " +  str(prob2)

		if datapoints[j][1] < params[0] + (params[1]*datapoints[j][0]):
			prob1 = classifier(i, params, datapoints[j])
		else:
			prob1 = 1 - classifier(i, params, datapoints[j])
		difference = prob2 - prob1

		differenceSquared = math.pow(difference,2)
		withinVariance += differenceSquared

	probs = []
	for point in datapoints:
		if point[1] < params[0] + (params[1]*point[0]):
			probs.append(classifier(i, params, point))
		else:
			probs.append(1- classifier(i,params, point))
	
	var =  math.pow(np.std(probs),2)
	betweenVariance = math.pow(var, 2)

	sum += (withinVariance/float(betweenVariance))

	withinVariance = 0
	for j in range(0, len(datapoints)-1):

		if datapoints[j+1][1] > params[0] + (params[1]*datapoints[j+1][0]):
			prob2 = classifier(i, params, datapoints[j+1])
		else:
			prob2 = 1- classifier(i, params, datapoints[j+1])

		if datapoints[j][1] > params[0] + (params[1]*datapoints[j][0]):
			prob1 = classifier(i, params, datapoints[j])
		else:
			prob1 = 1 - classifier(i, params, datapoints[j])
		difference = prob2 - prob1

		differenceSquared = math.pow(difference,2)
		withinVariance += differenceSquared

	probs = []
	for point in datapoints:
		if point[1] > params[0] + (params[1]*point[0]):
			probs.append(classifier(i, params, point))
		else:
			probs.append(1-classifier(i,params, point))
	
	var =  math.pow(np.std(probs),2)
	betweenVariance = math.pow(var, 2)

	sum += (withinVariance/float(betweenVariance))

	return sum

def getNumErrorsLinear(datapoints, params):
	leftbin = []
	rightbin = []
	for point in datapoints:
		if point[1] < params[0] + (params[1]*point[0]):
			leftbin.append(point)
		else:
			rightbin.append(point)

	# assume left is class zero

	class_errors_1 = 0
	for point in leftbin:
		if point[2]!=0:
			class_errors_1 += 1

	for point in rightbin:
		if point[2]!=1:
			class_errors_1 += 1

	# assume right class is zero
	class_errors_2 = 0
	for point in leftbin:
		if point[2]!=1:
			class_errors_2 += 1

	for point in rightbin:
		if point[2]!=0:
			class_errors_2 += 1

	# take min of both assumptions
	if class_errors_1 <= class_errors_2:
		return class_errors_1
	else:
		return class_errors_2


def getAllNeighbors(params):
	neighbors = []
	point1 = (params[0], params[1]+0.01)
	point2 = (params[0]+0.01, params[1])
	point3 = (params[0], params[1]-0.01)
	point4 = (params[0]-0.01, params[1])
	point5 = (params[0]+0.01, params[1]+0.01)
	point6 = (params[0]-0.01, params[1]-0.01)
	point7 = (params[0]-0.01, params[1]+0.01)
	point8 = (params[0]+0.01, params[1]-0.01)
	neighbors.append(point1)
	neighbors.append(point2)
	neighbors.append(point3)
	neighbors.append(point4)
	neighbors.append(point5)
	neighbors.append(point6)
	neighbors.append(point7)
	neighbors.append(point8)
	return neighbors

def hillClimb(params, datapoints, classifier):
	cost = costFunctionLinear(classifier, datapoints, params, 2)
	# print cost
	iterations = 0
	while True:
		iterations += 1
		neighbors = getAllNeighbors(params)
		changed = False

		for neighbor in neighbors:
			new_cost = costFunctionLinear(classifier, datapoints, neighbor, 2)
			if new_cost < cost:
				cost = new_cost
				params = neighbor
				changed = True

		# print params
		# print "---" 

		if changed==False or iterations > 1000:
			# print "iterations: " + str(iterations)
			break

	return params

# test linear prob
# thisClass = 1
# params = (3,0)
# point = (0,3)
# linearProbability(thisClass, params, point)

# sample from two 2D gaussians close to each other with segment length s
# x = []
# y = []
# datapoints = []
# s = 5 # segment length
# trials = 100

# while len(datapoints) < trials: #not exactly the trial number but whatever
# 	for i in range(0,s):
# 		newx = random.gauss(1,0.863)
# 		newy = random.gauss(0,1)
# 		x.append(newx)
# 		y.append(newy)
# 		datapoints.append((newx, newy, 0))

# 	for i in range(0,s):
# 		newx = random.gauss(1,0.863)
# 		newy = random.gauss(1,1)
# 		x.append(newx)
# 		y.append(newy)
# 		datapoints.append((newx, newy, 1))

# # plt.scatter(x,y)

# params = (0.5,0)
# first = getLine(params)
# print "# of datapoints: " + str(len(datapoints))
# # test cost function
# print "cost function min: " + str(costFunctionLinear(linearProbability, datapoints, params, 2))
# print "class_errors: " + str(getNumErrorsLinear(datapoints, params))
# print "success_rate: " + str(1-(getNumErrorsLinear(datapoints, params))/float(len(datapoints)))
# print "----"

# params = (2,0)
# second = getLine(params)
# print "cost function: " + str(costFunctionLinear(linearProbability, datapoints, params, 2))
# print "class_errors: " + str(getNumErrorsLinear(datapoints, params))
# print "----"

# params = (0,1)
# fifth = getLine(params)
# print "cost function: " + str(costFunctionLinear(linearProbability, datapoints, params, 2))
# print "class_errors: " + str(getNumErrorsLinear(datapoints, params))
# print "----"

# params = hillClimb((0,1), datapoints, linearProbability)
# sixth = getLine(params)
# print "hill climb optimum: " + str(costFunctionLinear(linearProbability, datapoints, params, 2))
# print "class_errors: " + str(getNumErrorsLinear(datapoints, params))
# print "success_rate: " + str(1-(getNumErrorsLinear(datapoints, params))/float(len(datapoints)))
# print "----"

# plt.plot(first[0], first[1])
# plt.plot(second[0], second[1])
# plt.plot(sixth[0], sixth[1])

# segmentLengths = [1,3,5,10,25]
# dataLengths = [50, 100, 500, 1000]

segmentLengths = [3,5,10,25]
dataLengths = [500, 1000]

for s in segmentLengths:
	for length in dataLengths:
		print "segment length: " + str(s) + " number of datapoints: " + str(length)
		success_rates = []
		for i in range(0,100):

			datapoints = []

			x1 = []
			y1 = []

			x2 = []
			y2 = []

			realprobs = []
			while len(datapoints) < length: #not exactly the dataLength but whatever
				for j in range(0,s):
					newx = random.gauss(1,0.863)
					newy = random.gauss(0,1)
					datapoints.append((newx, newy, 0))
					realprobs.append(0)
					x1.append(newx)
					y1.append(newy)

				for j in range(0,s):
					newx = random.gauss(1,0.863)
					newy = random.gauss(4,1)
					datapoints.append((newx, newy, 1))
					realprobs.append(1)
					x2.append(newx)
					y2.append(newy)

			params = hillClimb((0,1), datapoints, linearProbability)
			
			line = getLine(params)

			plt.plot(line[0], line[1], color="red", linewidth=2)
			plt.scatter(x1, y1, color="blue")
			plt.scatter(x2, y2, color="green")

			delta = 0.025
			x = np.arange(-3, 3, delta)
			y = np.arange(-2, 6, delta)
			X, Y = np.meshgrid(x, y)
			Z1 = mlab.bivariate_normal(X, Y, 0.863, 1, 1, 0)
			Z2 = mlab.bivariate_normal(X, Y, 0.863, 1, 1, 4)

			CS = plt.contour(X, Y, Z1)
			CS = plt.contour(X, Y, Z2)
			plt.clabel(CS, inline=1, fontsize = 10)


			plt.xlim(-2, 4)
			plt.show()


			success_rate = 1-((getNumErrorsLinear(datapoints, params))/float(len(datapoints)))
			success_rates.append(success_rate)
			print "success rate: " + str(success_rate)

			probs = getPosteriorProbs(params, datapoints)
			xs = range(len(probs))

			plt.figure()
			plt.plot(xs, probs, color="blue", linewidth = '2')
			plt.plot(xs, realprobs, color="green", linewidth= '2')
			plt.ylim(-0.5, 1.5)
			plt.xlim(0,100)

			plt.show()

		print "average success rate, 100 trials: " + str(np.mean(success_rates))
		print "success rate standard deviation, 100 trials: " + str(np.std(success_rates))

# segment length 3: 69.35% += 5.2%
# segment length 5: 70.00% += 5.8%

# plt.show()

# segment length 3, means 4 apart:
# average success rate, 100 trials: 0.975294117647
# success rate standard deviation, 100 trials: 0.0182089204176



