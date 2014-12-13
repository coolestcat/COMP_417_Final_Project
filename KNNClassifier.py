import random
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy.stats import multivariate_normal
import itertools
from sets import Set

print "Done Importing Stuff"
# parameters - each datapoint being in one of the n classes
# points: [x,y,trueClass,timeValue]

def getDistance2D(A,B):
	xy = math.pow((A[0]-B[0]),2) + math.pow((A[1]-B[1]),2)
	return math.sqrt(xy)

def getKNearestNeighbors(point, datapoints, k):
	pointDistances = []
	for p in datapoints:
		dist = getDistance2D(point, p)
		if dist!=0:
			pointDistances.append(([dist],p))

	pointDistances.sort()

	returnPoints = []
	for i in range(0,k):
		returnPoints.append(pointDistances[i][1])

	return returnPoints

def getKNNProb(neighborPoints, params, thisClass):
	count = 0
	for point in neighborPoints:
		if params[point[3]]==thisClass:
			count += 1

	return count/float(len(neighborPoints))

def getClassificationErrors(datapoints, params, nClasses):
	bins = []
	for i in range(0,nClasses):
		bins.append([])

	for i in range(len(params)):
		for j in range(0,nClasses):
			if params[i]==j:
				bins[j].append(i)

	errorPossibilities = []
	perms = itertools.permutations(range(0, nClasses))
	for perm in perms:
		ec = 0
		for j in range(0,nClasses):
			for elem in bins[j]:
				if datapoints[elem][2]!=perm[j]:
					ec += 1

		errorPossibilities.append(ec)

	# print errorPossibilities
	print "min errors: " + str(min(errorPossibilities))
	return min(errorPossibilities)

def costFunctionKNN(datapoints,params,nClasses,k):
	sum = 0
	for i in range(nClasses):
		withinVariance = 0
		for j in range(0, len(datapoints)-1):
			neighbors_jplus1 = getKNearestNeighbors(datapoints[j+1],datapoints,k)
			neighbors_j = getKNearestNeighbors(datapoints[j],datapoints,k)
			prob_jplus1 = getKNNProb(neighbors_jplus1, params, i)
			prob_j = getKNNProb(neighbors_j, params, i)
			differenceSquared = math.pow((prob_jplus1-prob_j),2)
			withinVariance += differenceSquared

		probs = []
		for point in datapoints:
			neighbors = getKNearestNeighbors(point, datapoints, k)
			prob = getKNNProb(neighbors, params, i)
			probs.append(prob)

		var = math.pow((np.std(probs)),2)
		betweenVariance = math.pow(var,2)

		sum += (withinVariance/float(betweenVariance))

	return sum

def getRandomNeighbor(params, numPointsChanged, datapoints, iterations, nClasses):
	newParams = []
	for elem in params:
		newParams.append(elem)
	
	if iterations < 500:
		realNumPointsChanged = numPointsChanged
	else:
		realNumPointsChanged = 3


	for i in range(0, realNumPointsChanged):
		toChange = random.randint(0,len(params)-1)
		if iterations < 500:
			numNeighborsToChange = 4	
		else:
			numNeighborsToChange = 1

		theseNeighbors = getKNearestNeighbors(datapoints[toChange], datapoints, numNeighborsToChange)

		thisClass = params[toChange]
		choiceList = []
		for i in range(0, thisClass):
			choiceList.append(i)

		for i in range(thisClass+1, nClasses):
			choiceList.append(i)

		# newClass = random.randint(0,nClasses-1)
		newClass = random.choice(choiceList)

		newParams[toChange] = newClass
		for neighbor in theseNeighbors:
			newParams[neighbor[3]]=newClass

	return newParams

# with simulated annealing
def randomGradientDescent(datapoints, params, nClasses, k):
	cost = costFunctionKNN(datapoints, params, nClasses, k)
	iterations = 0
	returnParams = []
	for elem in params:
		returnParams.append(elem)

	p = 0.8
	while iterations < 3000:
		iterations += 1

		if iterations%50 == 0:
			print iterations

		neighbor = getRandomNeighbor(returnParams, 10, datapoints, iterations, nClasses)
		neighborCost = costFunctionKNN(datapoints, neighbor, nClasses, k)

		p = p * 0.95
		if neighborCost < cost:
			cost = neighborCost
			print "cost changed:  " + str(cost) #  + " given by: " + str(neighbor)
			returnParams = []
			for elem in neighbor:
				returnParams.append(elem)
		else:
			decision = np.random.binomial(1, p)
			if decision == 1:
				cost = neighborCost
				print "cost changed:  " + str(cost) #  + " given by: " + str(neighbor)
				returnParams = []
				for elem in neighbor:
					returnParams.append(elem)
			else:
				continue

	return returnParams

# SUCCESSFUL RUN 1

# final params: [2, 2, 2, 1, 1, 1, 2, 0, 0, 2, 2, 2, 1, 1, 2, 0, 0, 0, 2, 1, 2, 1, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0]
# ideal cost: 899.172056871
# [0, 24, 24, 36, 36, 24]
# ideal errors: 0
# random cost: 13202.2689949
# [27, 29, 24, 20, 25, 19]
# random errors: 19
# after cost: 926.226612458
# [25, 34, 34, 25, 22, 4]
# after errors: 4

# SUCCESSFUL RUN 2

# final params: [1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2]
# ideal cost: 675.072243047
# [0, 24, 24, 36, 36, 24]
# ideal errors: 0
# random cost: 9235.61827456
# [28, 24, 20, 23, 21, 28]
# random errors: 20
# after cost: 669.124619676
# [25, 35, 1, 23, 24, 36]
# after errors: 1

# PERFECT RUN

# Ideal: [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]
# Random: [2, 1, 0, 2, 0, 2, 0, 0, 1, 2, 1, 0, 1, 1, 2, 1, 1, 0, 1, 2, 2, 1, 0, 1, 0, 0, 0, 0, 2, 1, 2, 0, 0, 1, 1, 1]
# Optimized: [2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0]
# ideal cost: 699.39522206
# [0, 24, 24, 36, 36, 24]
# ideal errors: 0
# random cost: 12716.7773251
# [29, 23, 28, 21, 22, 21]
# random errors: 21
# after cost: 699.39522206
# [24, 36, 36, 24, 24, 0]
# after errors: 0

# 3 Sparse Uniform Distributions

s = 3
trials = 36
success_rates = []

for i in range(0,3):
	print i
	datapoints = []
	c = 0

	while len(datapoints) < trials:
		square0x = np.random.uniform(low=-1,high=0, size=s)
		square0y = np.random.uniform(low=0,high=1, size=s)
		square0 = []
		for i in range(len(square0x)):
			square0.append([square0x[i], square0y[i]])

		square1x = np.random.uniform(low=-1,high=0, size=s)
		square1y = np.random.uniform(low=-1,high=0, size=s)
		square1 = []
		for i in range(len(square1x)):
			square1.append([square1x[i], square1y[i]])

		square2x = np.random.uniform(low=0,high=1, size=s)
		square2y = np.random.uniform(low=-0.5,high=0.5, size=s)
		square2 = []
		for i in range(len(square2x)):
			square2.append([square2x[i], square2y[i]])

		for point in square0:
			datapoints.append([point[0], point[1], 0, c])
			c += 1

		for point in square1:
			datapoints.append([point[0], point[1], 1, c])
			c += 1

		for point in square2:
			datapoints.append([point[0], point[1], 2, c])
			c += 1

	randomparams = []
	for i in range(len(datapoints)):
		randomparams.append(random.randint(0,2))

	afterparams = randomGradientDescent(datapoints, randomparams, 3, 10)
	afterCost = costFunctionKNN(datapoints, afterparams, 3, 10)
	success_rate = 1-(getClassificationErrors(datapoints, afterparams, 3)/float(len(datapoints)))
	print "success rate: " + str(success_rate)
	success_rates.append(success_rate)

	# PLOT 
	bins = []
	for i in range(0,3):
		bins.append([])

	for i in range(len(datapoints)): 
		for j in range(0,3):
			if afterparams[i] == j:
				bins[j].append(datapoints[i])

	x = []
	for i in range(0,3):
		inner = []
		for j in range(0,3):
			inner.append([])
		x.append(inner)

	y = []
	for i in range(0,3):
		inner = []
		for j in range(0,3):
			inner.append([])
		y.append(inner)

	for i in range(0,3):
		for point in bins[i]:
			for k in range(0,3):
				if point[2] == k:
					x[i][k].append(point[0])
					y[i][k].append(point[1])

	colors = ['red', 'green', 'blue']
	markers = ['o', '^', 'x']

	for i in range(0,3):
		for j in range(0,3):
			plt.scatter(x[i][j], y[i][j], color = colors[j], marker = markers[i])

	plt.show()



print "average success rate sparse, 100 trials: " + str(np.mean(success_rates))
print "standard deviation of success rate sparse, 100 trials: " + str(np.std(success_rates))

# 3 Uniform Distributions of Unequal Size

s = 3
trials = 200
success_rates = []

for i in range(0,50):
	print i
	datapoints = []
	c = 0

	while len(datapoints) < trials:
		square0x = np.random.uniform(low=-1,high=0, size=s)
		square0y = np.random.uniform(low=0,high=1, size=s)
		square0 = []
		for i in range(len(square0x)):
			square0.append([square0x[i], square0y[i]])

		square1x = np.random.uniform(low=0,high=2, size=2*s)
		square1y = np.random.uniform(low=0,high=1, size=2*s)
		square1 = []
		for i in range(len(square1x)):
			square1.append([square1x[i], square1y[i]])

		square2x = np.random.uniform(low=-1,high=2, size=3*s)
		square2y = np.random.uniform(low=-1,high=0, size=3*s)
		square2 = []
		for i in range(len(square2x)):
			square2.append([square2x[i], square2y[i]])

		for point in square0:
			datapoints.append([point[0], point[1], 0, c])
			c += 1

		for point in square1:
			datapoints.append([point[0], point[1], 1, c])
			c += 1

		for point in square2:
			datapoints.append([point[0], point[1], 2, c])
			c += 1

	# print "# of datapoints: " + str(len(datapoints))

	idealparams = []
	for point in datapoints:
		idealparams.append(point[2])

	# print "ideal cost: " + str(costFunctionKNN(datapoints, idealparams, 3, 20))

	randomparams = []
	for i in range(len(datapoints)):
		randomparams.append(random.randint(0,2))

	afterparams = randomGradientDescent(datapoints, randomparams, 3, 20)
	afterCost = costFunctionKNN(datapoints, afterparams, 3, 20)
	success_rate = 1-(getClassificationErrors(datapoints, afterparams, 3)/float(len(datapoints)))
	print "success rate: " + str(success_rate)
	success_rates.append(success_rate)

	# PLOT 
	bins = []
	for i in range(0,3):
		bins.append([])

	for i in range(len(datapoints)): 
		for j in range(0,3):
			if afterparams[i] == j:
				bins[j].append(datapoints[i])

	x = []
	for i in range(0,3):
		inner = []
		for j in range(0,3):
			inner.append([])
		x.append(inner)

	y = []
	for i in range(0,3):
		inner = []
		for j in range(0,3):
			inner.append([])
		y.append(inner)

	for i in range(0,3):
		for point in bins[i]:
			for k in range(0,3):
				if point[2] == k:
					x[i][k].append(point[0])
					y[i][k].append(point[1])

	colors = ['red', 'green', 'blue']
	markers = ['o', '^', 'x']

	for i in range(0,3):
		for j in range(0,3):
			plt.scatter(x[i][j], y[i][j], color = colors[j], marker = markers[i])

	plt.show()


print "average success rate, 100 trials: " + str(np.mean(success_rates))
print "standard deviation of success rate, 100 trials: " + str(np.std(success_rates))

# 2 Overlapping Uniform Distributions

s = 3
trials = 200
success_rates = []

for i in range(0,50):
	print i
	datapoints = []
	x = []
	y = []
	c = 0

	while len(datapoints) < trials:
		square0x = np.random.uniform(low=0,high=1, size=s)
		square0y = np.random.uniform(low=0,high=1, size=s)
		square0 = []
		for i in range(len(square0x)):
			square0.append([square0x[i], square0y[i]])

		square1x = np.random.uniform(low=0.6,high=1.6, size=s)
		square1y = np.random.uniform(low=0,high=1, size=s)
		square1 = []
		for i in range(len(square1x)):
			square1.append([square1x[i], square1y[i]])

		for point in square0:
			datapoints.append([point[0], point[1], 0, c])
			x.append(point[0])
			y.append(point[1])
			c += 1

		for point in square1:
			datapoints.append([point[0], point[1], 1, c])
			x.append(point[0])
			y.append(point[1])
			c += 1

	# plt.scatter(x,y)
	# plt.show()

	# print "# of datapoints: " + str(len(datapoints))

	idealparams = []
	for point in datapoints:
		idealparams.append(point[2])

	print "ideal cost: " + str(costFunctionKNN(datapoints, idealparams, 2, 20))

	randomparams = []
	for i in range(len(datapoints)):
		randomparams.append(random.randint(0,1))

	afterparams = randomGradientDescent(datapoints, randomparams, 2, 20)
	afterCost = costFunctionKNN(datapoints, afterparams, 2, 20)
	print "after cost: " + str(afterCost)
	success_rate = 1-(getClassificationErrors(datapoints, afterparams, 2)/float(len(datapoints)))
	print "success rate: " + str(success_rate)
	success_rates.append(success_rate)

	bins = []
	for i in range(0,2):
		bins.append([])

	for i in range(len(datapoints)): 
		for j in range(0,2):
			if afterparams[i] == j:
				bins[j].append(datapoints[i])

	x = []
	for i in range(0,2):
		inner = []
		for j in range(0,2):
			inner.append([])
		x.append(inner)

	y = []
	for i in range(0,2):
		inner = []
		for j in range(0,2):
			inner.append([])
		y.append(inner)

	for i in range(0,2):
		for point in bins[i]:
			for k in range(0,2):
				if point[2] == k:
					x[i][k].append(point[0])
					y[i][k].append(point[1])

	colors = ['red', 'blue']
	markers = ['o', 'x']

	for i in range(0,2):
		for j in range(0,2):
			plt.scatter(x[i][j], y[i][j], color = colors[j], marker = markers[i])

	plt.show()

print "average success rate, 100 trials: " + str(np.mean(success_rates))
print "standard deviation of success rate, 100 trials: " + str(np.std(success_rates))

# 6 Uniform Distributions

s = 3
trials = 120
success_rates = []

for i in range(0,1):
	print i
	datapoints = []
	x = []
	y = []
	c = 0

	while len(datapoints) < trials:
		square0x = np.random.uniform(low=-1,high=0, size=s)
		square0y = np.random.uniform(low=0,high=1, size=s)
		square0 = []
		for i in range(len(square0x)):
			square0.append([square0x[i], square0y[i]])

		square1x = np.random.uniform(low=0,high=1, size=s)
		square1y = np.random.uniform(low=0,high=1, size=s)
		square1 = []
		for i in range(len(square1x)):
			square1.append([square1x[i], square1y[i]])

		square2x = np.random.uniform(low=1,high=2, size=s)
		square2y = np.random.uniform(low=0,high=1, size=s)
		square2 = []
		for i in range(len(square2x)):
			square2.append([square2x[i], square2y[i]])
		square3x = np.random.uniform(low=-1,high=0, size=s)
		square3y = np.random.uniform(low=-1,high=0, size=s)
		square3 = []
		for i in range(len(square3x)):
			square3.append([square3x[i], square3y[i]])

		square4x = np.random.uniform(low=0,high=1, size=s)
		square4y = np.random.uniform(low=-1,high=0, size=s)
		square4 = []
		for i in range(len(square4x)):
			square4.append([square4x[i], square4y[i]])

		square5x = np.random.uniform(low=1,high=2, size=s)
		square5y = np.random.uniform(low=-1,high=0, size=s)
		square5 = []
		for i in range(len(square5x)):
			square5.append([square5x[i], square5y[i]])

		for point in square0:
			datapoints.append([point[0], point[1], 0, c])
			x.append(point[0])
			y.append(point[1])
			c += 1

		for point in square1:
			datapoints.append([point[0], point[1], 1, c])
			x.append(point[0])
			y.append(point[1])
			c += 1

		for point in square2:
			datapoints.append([point[0], point[1], 2, c])
			x.append(point[0])
			y.append(point[1])
			c += 1

		for point in square3:
			datapoints.append([point[0], point[1], 3, c])
			x.append(point[0])
			y.append(point[1])
			c += 1

		for point in square4:
			datapoints.append([point[0], point[1], 4, c])
			x.append(point[0])
			y.append(point[1])
			c += 1

		for point in square5:
			datapoints.append([point[0], point[1], 5, c])
			x.append(point[0])
			y.append(point[1])
			c += 1

	plt.scatter(x,y)
	plt.show()

	print "# of datapoints: " + str(len(datapoints))

	idealparams = []
	for point in datapoints:
		idealparams.append(point[2])

	print "ideal cost: " + str(costFunctionKNN(datapoints, idealparams, 6, 10))
	print "ideal class errors: " + str(getClassificationErrors(datapoints, idealparams, 6))

	randomparams = []
	for i in range(len(datapoints)):
		randomparams.append(random.randint(0,5))

	afterparams = randomGradientDescent(datapoints, randomparams, 6, 10)
	afterCost = costFunctionKNN(datapoints, afterparams, 6, 10)
	print "after cost: " + str(afterCost)
	success_rate = 1-(getClassificationErrors(datapoints, afterparams, 6)/float(len(datapoints)))
	print "success rate: " + str(success_rate)
	success_rates.append(success_rate)

print "average success rate, 100 trials: " + str(np.mean(success_rates))
print "standard deviation of success rate, 100 trials: " + str(np.std(success_rates))

# Two Spirals

s = 3
trials = 1000
success_rates = []

for i in range(0,1):
	print i
	datapoints = []

	x1 = []
	y1 = []

	x2 = []
	y2 = []
	c = 0

	while len(datapoints) < trials:
		darc = 15
		sigma = 0.9


		arm0 = []
		for i in range(s):
			uniformRandom = random.uniform(0,1)
			phi = darc * math.sqrt(uniformRandom)
			dperp = random.gauss(0, sigma)
			tt = phi
			x = (tt + dperp)*math.cos(phi)
			y = (tt + dperp)*math.sin(phi)
			arm0.append([x,y])

		arm1 = []
		for i in range(s):
			tt  = 5
			uniformRandom = random.uniform(0,1)
			phi = darc * math.sqrt(uniformRandom)
			dperp = random.gauss(0, sigma)
			tt = phi
			x = (tt + dperp)*math.cos(phi + math.pi)
			y = (tt + dperp)*math.sin(phi + math.pi)
			arm1.append([x,y])

		for point in arm0:
			datapoints.append([point[0], point[1], 0, c])
			x1.append(point[0])
			y1.append(point[1])
			c += 1

		for point in arm1:
			datapoints.append([point[0], point[1], 1, c])
			x2.append(point[0])
			y2.append(point[1])
			c += 1

	plt.scatter(x1,y1, color="blue")
	plt.scatter(x2,y2, color="red")
	plt.show()

	# print "# of datapoints: " + str(len(datapoints))

	idealparams = []
	for point in datapoints:
		idealparams.append(point[2])

	print "ideal cost: " + str(costFunctionKNN(datapoints, idealparams, 2, 20))

	randomparams = []
	for i in range(len(datapoints)):
		randomparams.append(random.randint(0,1))

	afterparams = randomGradientDescent(datapoints, randomparams, 2, 20)
	afterCost = costFunctionKNN(datapoints, afterparams, 2, 20)
	print "after cost: " + str(afterCost)
	success_rate = 1-(getClassificationErrors(datapoints, afterparams, 2)/float(len(datapoints)))
	print "success rate: " + str(success_rate)
	success_rates.append(success_rate)

	bins = []
	for i in range(0,2):
		bins.append([])

	for i in range(len(datapoints)): 
		for j in range(0,2):
			if afterparams[i] == j:
				bins[j].append(datapoints[i])

	x = []
	for i in range(0,2):
		inner = []
		for j in range(0,2):
			inner.append([])
		x.append(inner)

	y = []
	for i in range(0,2):
		inner = []
		for j in range(0,2):
			inner.append([])
		y.append(inner)

	for i in range(0,2):
		for point in bins[i]:
			for k in range(0,2):
				if point[2] == k:
					x[i][k].append(point[0])
					y[i][k].append(point[1])

	colors = ['red', 'blue']
	markers = ['o', 'x']
	plt.figure()
	for i in range(0,2):
		for j in range(0,2):
			plt.scatter(x[i][j], y[i][j], color = colors[i], marker = 'o')

	plt.show()

print "average success rate, 100 trials: " + str(np.mean(success_rates))
print "standard deviation of success rate, 100 trials: " + str(np.std(success_rates))





