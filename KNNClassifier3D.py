import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import sys
from scipy.stats import multivariate_normal
import itertools
from sets import Set
from sklearn.cluster import MiniBatchKMeans, KMeans

print "Done Importing Stuff"
# parameters - each datapoint being in one of the n classes
# points: [x,y,z,trueClass,timeValue]

def getDistance3D(A,B):
	xy = math.pow((A[0]-B[0]),2) + math.pow((A[1]-B[1]),2) + math.pow((A[2]-B[2]),2)
	return math.sqrt(xy)

def getKNearestNeighbors(point, datapoints, k):
	pointDistances = []
	for p in datapoints:
		dist = getDistance3D(point, p)
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
		if params[point[4]]==thisClass:
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
				if datapoints[elem][3]!=perm[j]:
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

	if iterations < 2000:
		realNumPointsChanged = numPointsChanged
	else:
		realNumPointsChanged = 1

	for i in range(0, realNumPointsChanged):
		toChange = random.randint(0,len(params)-1)

		if iterations < 2000:
			numNeighborsToChange = 2
		else:
			numNeighborsToChange = 0

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

def randomGradientDescent(datapoints, params, nClasses, k):
	cost = costFunctionKNN(datapoints, params, nClasses, k)
	iterations = 0
	returnParams = []
	for elem in params:
		returnParams.append(elem)

	p = 0.5
	while iterations < 3000:
		iterations += 1

		if iterations%50 == 0:
			print iterations

		neighbor = getRandomNeighbor(returnParams, 3, datapoints, iterations, nClasses)
		neighborCost = costFunctionKNN(datapoints, neighbor, nClasses, k)

		p = p * 0.95
		if neighborCost < cost:
			cost = neighborCost
			# print "cost changed:  " + str(cost) #  + " given by: " + str(neighbor)
			returnParams = []
			for elem in neighbor:
				returnParams.append(elem)
		else:
			decision = np.random.binomial(1, p)
			if decision == 1:
				cost = neighborCost
			# 	print "cost changed:  " + str(cost) #  + " given by: " + str(neighbor)
				returnParams = []
				for elem in neighbor:
					returnParams.append(elem)
			else:
				continue

	return returnParams

s = 3
trials = 36
success_rates = []
kmeans_success_rates = []

for i in range(0,100):
	print i
	datapoints = []
	justDatapoints = []

	x1 = []
	y1 = []
	z1 = []

	x2 = []
	y2 = []
	z2 = []

	x3 = []
	y3 = []
	z3 = []

	c = 0

	while len(datapoints) < trials:
		square0x = np.random.uniform(low=-1,high=0, size=s)
		square0y = np.random.uniform(low=0,high=1, size=s)
		square0z = np.random.uniform(low=0,high=1, size=s)
		square0 = []
		for i in range(len(square0x)):
			square0.append([square0x[i], square0y[i], square0z[i]])

		square1x = np.random.uniform(low=-1,high=0, size=s)
		square1y = np.random.uniform(low=-1,high=0, size=s)
		square1z = np.random.uniform(low=0,high=1, size=s)
		square1 = []
		for i in range(len(square1x)):
			square1.append([square1x[i], square1y[i], square1z[i]])

		square2x = np.random.uniform(low=0,high=1, size=s)
		square2y = np.random.uniform(low=-0.5,high=0.5, size=s)
		square2z = np.random.uniform(low=0,high=1, size=s)
		square2 = []
		for i in range(len(square2x)):
			square2.append([square2x[i], square2y[i], square2z[i]])

		for point in square0:
			datapoints.append([point[0], point[1], point[2], 0, c])
			justDatapoints.append([point[0], point[1], point[2]])
			x1.append(point[0])
			y1.append(point[1])
			z1.append(point[2])
			c += 1

		for point in square1:
			datapoints.append([point[0], point[1], point[2], 1, c])
			justDatapoints.append([point[0], point[1], point[2]])
			x2.append(point[0])
			y2.append(point[1])
			z2.append(point[2])
			c += 1

		for point in square2:
			datapoints.append([point[0], point[1], point[2], 2, c])
			justDatapoints.append([point[0], point[1], point[2]])
			x3.append(point[0])
			y3.append(point[1])
			z3.append(point[2])
			c += 1



	idealparams = []
	for i in range(len(datapoints)):
		idealparams.append(datapoints[i][3])

	randomparams = []
	for i in range(len(datapoints)):
		randomparams.append(random.randint(0,2))

	idealCost = costFunctionKNN(datapoints, idealparams, 3, 10)
	# print "ideal cost: " + str(idealCost)

	afterparams = randomGradientDescent(datapoints, randomparams, 3, 10)

	afterCost = costFunctionKNN(datapoints, afterparams, 3, 10)
	# print "after cost: " + str(afterCost)

	success_rate = 1-(getClassificationErrors(datapoints, afterparams, 3)/float(len(datapoints)))
	print "success rate: " + str(success_rate)
	success_rates.append(success_rate)

	# compare with k-means
	k_means = KMeans(init='k-means++', n_clusters=3)
	predicted = k_means.fit_predict(justDatapoints)
	success_rate = 1-((getClassificationErrors(datapoints, predicted, 3))/float(len(datapoints)))
	print "kmeans success rate: " + str(success_rate)
	kmeans_success_rates.append(success_rate)
	print "---"

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

	z = []
	for i in range(0,3):
		inner = []
		for j in range(0,3):
			inner.append([])
		z.append(inner)


	for i in range(0,3):
		for point in bins[i]:
			for k in range(0,3):
				if point[3] == k:
					x[i][k].append(point[0])
					y[i][k].append(point[1])
					z[i][k].append(point[2])

	colors = ['red', 'green', 'blue']
	markers = ['o', '^', 'x']

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for i in range(0,3):
		for j in range(0,3):
			ax.scatter(x[i][j], y[i][j], z[i][j], color = colors[j], marker = markers[i])

	plt.show()


print "average success rate, 100 trials: " + str(np.mean(success_rates))
print "standard deviation of success rate, 100 trials: " + str(np.std(success_rates))