import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import numpy as np
import math
import sys
from scipy.stats import multivariate_normal
import itertools
from sets import Set
from sklearn.cluster import MiniBatchKMeans, KMeans

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


print "Finished Importing Stuff"

point0 = [0.95, 0]
point120 = [-0.95/float(2), (0.95/float(2))*math.sqrt(3.0)]
point240 = [-0.95/float(2), -1*(0.95/float(2))*math.sqrt(3.0)]
symmetriccov = [[1,0],[0,1]]
cov0 = [[0.5,0],[0,1]]
cov120 = [[0.8, 0.1], [0.1, 0.6]]
cov240 = [[0.8, -0.1], [-0.1, 0.6]]

print point0
print point120
print point240


# gaussian pdf 
def getGaussianPdf(mean, cov):
	return multivariate_normal(mean = mean, cov=[[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]])

def costFunctionGaussianResponses(datapoints, params, nClasses):
	responses = np.zeros((nClasses,len(datapoints)))

	var = []
	for j in range(nClasses):
		var.append(getGaussianPdf(params[j][0], params[j][1]))

	for j in range(len(datapoints)):
		p = [datapoints[j][0], datapoints[j][1]]
		for i in range(nClasses):
			responses[i,j] = (1/float(nClasses)) * var[i].pdf(p)

	responses = responses / np.sum(responses,axis=0)

	sum = 0
	for i in range(nClasses):
		withinVariance = 0
		for j in range(0, len(datapoints)-1):
			difference = responses[i,j+1] - responses[i,j]
			differenceSquared = math.pow(difference, 2)
			withinVariance += differenceSquared

		v = math.pow(np.std(responses[i]), 2)
		betweenVariance = math.pow(v,2)

		sum += (withinVariance/float(betweenVariance))

	return sum


def getNumErrorsGaussian2(datapoints, params):
	print params
	var0 = getGaussianPdf(params[0][0], params[0][1])
	var1 = getGaussianPdf(params[1][0], params[1][1])

	bin0 = []
	bin1 = []

	for point in datapoints:
		p = [point[0], point[1]]
		v0 = var0.pdf(p)
		v1 = var1.pdf(p)

		if max(v0, v1) == v0:
			bin0.append(point)
		else:
			bin1.append(point)

	errorPossibilities = []
	perms = itertools.permutations([0,1])
	for perm in perms: #try each permutation of data labels
		ce = 0
		for point in bin0:
			if point[2]!=perm[0]:
				ce += 1

		for point in bin1:
			if point[2]!=perm[1]:
				ce += 1

		errorPossibilities.append(ce)

	print errorPossibilities
	return min(errorPossibilities)

def getNumErrorsKmeans(datapoints, predicted, nClasses):
	bins = []
	for i in range(0,nClasses):
		bins.append([])

	for i in range(len(datapoints)):
		for j in range(0,nClasses):
			if predicted[i]==j:
				bins[j].append(datapoints[i])

	errorPossibilities = []
	perms = itertools.permutations(range(0, nClasses))
	for perm in perms:
		ec = 0
		for j in range(0,nClasses):
			for elem in bins[j]:
				if elem[2]!=perm[j]:
					ec += 1

		errorPossibilities.append(ec)

	print errorPossibilities
	return min(errorPossibilities)

def getNumErrorsGaussian(datapoints, params):
	var0 = getGaussianPdf(params[0][0], params[0][1])
	var1 = getGaussianPdf(params[1][0], params[1][1])
	var2 = getGaussianPdf(params[2][0], params[2][1])

	bin0 = []
	bin1 = []
	bin2 = []

	for point in datapoints:
		p = [point[0], point[1]]
		v0 = var0.pdf(p)
		v1 = var1.pdf(p)
		v2 = var2.pdf(p)
		if max(v0, v1, v2) == v0:
			bin0.append(point)
		elif max(v0, v1, v2)==v1:
			bin1.append(point)
		else:
			bin2.append(point)

	errorPossibilities = []
	perms = itertools.permutations([0,1,2])
	for perm in perms: #try each permutation of data labels
		ce = 0
		for point in bin0:
			if point[2]!=perm[0]:
				ce += 1

		for point in bin1:
			if point[2]!=perm[1]:
				ce += 1

		for point in bin2:
			if point[2]!=perm[2]:
				ce += 1

		errorPossibilities.append(ce)

	print errorPossibilities
	return min(errorPossibilities)

def getAllNeighborsWithCov(params):
	neighbors = []
	additions = [-0.1, 0, +0.1]
	for i in range(0, 3):
		for j in range(0, 3):
			for k in range(0, 3):
				for l in range(0,3):
					for m in range(0,3):
						for n in range(0,3):
							for o in range(0,3):
								for p in range(0,3):
									for q in range(0,3):
										mean1 = [params[0][0][0] + additions[i], params[0][0][1] + additions[j]]
										mean2 = [params[1][0][0] + additions[k], params[1][0][1] + additions[l]]
										mean3 = [params[2][0][0] + additions[m], params[2][0][1] + additions[n]]
										cov1 = [[params[0][1][0][0] + additions[o],0],[0,params[0][1][0][0] + additions[o]]]
										cov2 = [[params[1][1][0][0] + additions[p],0],[0,params[1][1][0][0] + additions[p]]]
										cov3 = [[params[2][1][0][0] + additions[q],0],[0,params[2][1][0][0] + additions[q]]]
										neighbor = [[mean1, cov1],[mean2, cov2],[mean3, cov3]]
										neighbors.append(neighbor)

	return neighbors

# for 2 Gaussians instead of 3
def getAllNeighbors2(params):
	neighbors = []
	additions = [-0.1, 0, +0.1]
	for i in range(0, 3):
		for j in range(0, 3):
			for k in range(0, 3):
				for l in range(0,3):
					mean1 = [params[0][0][0] + additions[i], params[0][0][1] + additions[j]]
					mean2 = [params[1][0][0] + additions[k], params[1][0][1] + additions[l]]
					symmetriccov = [[1,0],[0,1]]
					cov1 = cov2 = symmetriccov
					neighbor = [[mean1, cov1], [mean2, cov2]]
					neighbors.append(neighbor)

	return neighbors

# assume radially symmetric gaussians
def getAllNeighbors(params):
	neighbors = []
	additions = [-0.1, 0, +0.1]
	for i in range(0, 3):
		for j in range(0, 3):
			for k in range(0, 3):
				for l in range(0,3):
					for m in range(0,3):
						for n in range(0,3):
							mean1 = [params[0][0][0] + additions[i], params[0][0][1] + additions[j]]
							mean2 = [params[1][0][0] + additions[k], params[1][0][1] + additions[l]]
							mean3 = [params[2][0][0] + additions[m], params[2][0][1] + additions[n]]
							symmetriccov = [[1,0],[0,1]]
							cov1 = cov2 = cov3 = symmetriccov
							neighbor = [[mean1, cov1],[mean2, cov2],[mean3, cov3]]
							neighbors.append(neighbor)

	return neighbors

def gradientDescent(params, datapoints, nClasses):
	cost = costFunctionGaussianResponses(datapoints, params, nClasses)

	iterations = 0
	while True:
		# print "cost: " + str(cost)
		# print "params: " + str(params)

		# FIND BEST NEIGHBOR

		# print "cost: " + str(cost)
		# iterations += 1
		# if nClasses == 3:
		# 	neighbors = getAllNeighbors(params)
		# else:
		# 	neighbors = getAllNeighbors2(params)
		# changed = False

		# for neighbor in neighbors:
		# 	new_cost = costFunctionGaussianResponses(datapoints, neighbor, nClasses)
		# 	if new_cost < cost:
		# 		cost = new_cost
		# 		params = neighbor
		# 		changed = True

		# if changed==False or iterations > 10:
		# 	break


		# print iterations

		# RANDOM NEIGHBOR THAT HAS IMPROVEMENT

		iterations += 1
		if nClasses == 3:
			neighbors = getAllNeighbors(params)
		else:
			neighbors = getAllNeighbors2(params)
		changed = False

		r = random.randint(0, len(neighbors)-1)
		new_cost = costFunctionGaussianResponses(datapoints, neighbors[r], nClasses)
		if new_cost < cost:
			if math.fabs(cost - new_cost) > 1:
				cost = new_cost
				params = neighbors[r]
				changed = True
			#else changed = False and we break
		else:
			if iterations > 100000:
				break
			continue

		if changed==False or iterations > 100000:
			break

	return params

params = [(point0, cov0), (point120, cov120), (point240, cov240)]
nClasses = 3
print len(getAllNeighbors(params))

datapoints = []
x = []
y = []
s = 3
trials = 200

while len(datapoints) < trials:
	samples0 = np.random.multivariate_normal([0.95,0], [[0.5, 0],[0, 1]], s)
	samples120 = np.random.multivariate_normal(point120, cov120, s)
	samples240 = np.random.multivariate_normal(point240, cov240, s)

	for sample in samples0:
		x.append(sample[0])
		y.append(sample[1])
		datapoints.append((sample[0],sample[1],0))

	for sample in samples120:
		x.append(sample[0])
		y.append(sample[1])
		datapoints.append((sample[0],sample[1],120))

	for sample in samples240:
		x.append(sample[0])
		y.append(sample[1])	
		datapoints.append((sample[0],sample[1],240))

# plt.scatter(x,y)
# plt.show()


datapoints2D = []
x = []
y = []
s = 3
trials = 200
while len(datapoints2D) < trials:
	for i in range(0,s):
		newx = random.gauss(1,1)
		newy = random.gauss(0,1)
		x.append(newx)
		y.append(newy)
		datapoints2D.append((newx, newy, 0))

	for i in range(0,s):
		newx = random.gauss(1,1)
		newy = random.gauss(2,1)
		x.append(newx)
		y.append(newy)
		datapoints2D.append((newx, newy, 1))

# plt.scatter(x,y)
# plt.show()

# params = [([1,0], symmetriccov), ([1,2], symmetriccov)]
# cost = costFunctionGaussianResponses(datapoints2D, params, 2)
# print "min cost 2 gaussians: " + str(cost)
# errors = getNumErrorsGaussian2(datapoints2D, params)
# print "class_errors: " + str(errors)
# print "success_rate: " + str(1-(errors/float(len(datapoints2D))))
# print "---"

# startp = [([0,0], symmetriccov), ([1,0], symmetriccov)]
# params = gradientDescent(startp, datapoints2D, 2)
# cost = costFunctionGaussianResponses(datapoints2D, params, 2)
# print "gradient descent min cost 2 gaussians: " + str(cost)
# errors = getNumErrorsGaussian2(datapoints2D, params)
# print "class_errors: " + str(errors)
# print "success_rate: " + str(1-(errors/float(len(datapoints2D))))
# print "---"

success_rates = []
optimal_success_rates = []
startp = [([0,0], symmetriccov), ([0,2], symmetriccov)]
x1 = []
y1 = []

x2 = []
y2 = []

for i in range(0,100):
	print i
	datapoints2D = []

	while len(datapoints2D) < trials: #not exactly the trial number but whatever
		for i in range(0,s):
			newx = random.gauss(1,1)
			newy = random.gauss(0,1)
			x1.append(newx)
			y1.append(newy)
			datapoints2D.append((newx, newy, 0))

		for i in range(0,s):
			newx = random.gauss(1,1)
			newy = random.gauss(2,1)
			x2.append(newx)
			y2.append(newy)
			datapoints2D.append((newx, newy, 1))

	params = gradientDescent(startp, datapoints2D, 2)
	success_rate = 1-((getNumErrorsGaussian2(datapoints2D, params))/float(len(datapoints)))
	success_rates.append(success_rate)
	print "success_rate: " + str(success_rate)
	optimalparams = [([1,0], symmetriccov), ([1,2], symmetriccov)]
	optimal_success_rate = 1-((getNumErrorsGaussian2(datapoints2D, optimalparams))/float(len(datapoints)))
	optimal_success_rates.append(optimal_success_rate)

	# plot

	plt.figure()
	plt.scatter(x1,y1, color='red')
	plt.scatter(x2,y2, color = 'blue')

	delta = 0.025
	x = np.arange(-2, 4, delta)
	y = np.arange(-4, 4, delta)
	X, Y = np.meshgrid(x, y)
	Z1 = mlab.bivariate_normal(X, Y, 1, 1, params[0][0][0], params[0][0][1])
	Z2 = mlab.bivariate_normal(X, Y, 1, 1, params[1][0][0], params[1][0][1])

	CS = plt.contour(X, Y, Z1)
	CS = plt.contour(X, Y, Z2)
	plt.clabel(CS, inline=1, fontsize = 10)
	plt.title('Gaussians Obtained from Algorithm')

	plt.xlim(-2,4)
	plt.ylim(-4,4)
	plt.show()

print "average optimal success rate 2 gaussians, 100 trials: " + str(np.mean(optimal_success_rates))
print "optimal success rate standard deviation 2 gaussians, 100 trials: " + str(np.std(optimal_success_rates))
print "average success rate 2 gaussians, 100 trials: " + str(np.mean(success_rates))
print "success rate standard deviation 2 gaussians, 100 trials: " + str(np.std(success_rates))

print "------------------------------"

# params = [(point0, symmetriccov), (point120, symmetriccov), (point240, symmetriccov)]
# cost = costFunctionGaussianResponses(datapoints, params, nClasses)
# print "min cost: " + str(cost)
# errors = getNumErrorsGaussian(datapoints, params)
# print "class_errors: " + str(errors)
# print "success_rate: " + str(1-(errors/float(len(datapoints))))
# print "---"

# params = [(point0, cov0), (point120, cov120), (point240, cov240)]
# cost = costFunctionGaussianResponses(datapoints, params, nClasses)
# print "min cost: " + str(cost)
# errors = getNumErrorsGaussian(datapoints, params)
# print "class_errors: " + str(errors)
# print "success_rate: " + str(1-(errors/float(len(datapoints))))
# print "---"

# params = [([1,0], symmetriccov), ([0,1], symmetriccov), ([-1,0], symmetriccov)]
# cost = costFunctionGaussianResponses(datapoints, params, nClasses)
# print "min cost: " + str(cost)
# errors = getNumErrorsGaussian(datapoints, params)
# print "class_errors: " + str(errors)
# print "success_rate: " + str(1-(errors/float(len(datapoints))))
# print "---"

startp = [([0,0], symmetriccov), ([0,0], symmetriccov), ([0,0], symmetriccov)]
# startp = [([1,0], symmetriccov), ([0,1], symmetriccov), ([-1,0], symmetriccov)]
success_rates = []
kmeans_success_rates = []
for i in range(0,100):
	print i
	datapoints = []
	justDatapoints = []
	x1 = []
	y1 = []

	x2 = []
	y2 = []

	x3 = []
	y3 = []

	while len(datapoints) < trials: #not exactly the trial number but whatever
		samples0 = np.random.multivariate_normal([0.95,0], [[0.5, 0],[0, 1]], s)
		samples120 = np.random.multivariate_normal(point120, cov120, s)
		samples240 = np.random.multivariate_normal(point240, cov240, s)

		for sample in samples0:
			datapoints.append((sample[0],sample[1],0))
			justDatapoints.append([sample[0], sample[1]])
			x1.append(sample[0])
			y1.append(sample[1])

		for sample in samples120:
			datapoints.append((sample[0],sample[1],1))
			justDatapoints.append([sample[0], sample[1]])
			x2.append(sample[0])
			y2.append(sample[1])

		for sample in samples240:
			datapoints.append((sample[0],sample[1],2))
			justDatapoints.append([sample[0], sample[1]])
			x3.append(sample[0])
			y3.append(sample[1])

	params = gradientDescent(startp, datapoints, 3)
	success_rate = 1-((getNumErrorsGaussian(datapoints, params))/float(len(datapoints)))
	print "success_rate: " + str(success_rate)
	# print params
	success_rates.append(success_rate)

	# compare with kmeans:
	k_means = KMeans(init='k-means++', n_clusters=3)
	predicted = k_means.fit_predict(justDatapoints)
	success_rate = 1-((getNumErrorsKmeans(datapoints, predicted, 3))/float(len(datapoints)))
	print "kmeans success rate: " + str(success_rate)
	kmeans_success_rates.append(success_rate)

	# plot
	# plt.figure()
	# plt.scatter(x1,y1, color='red')
	# plt.scatter(x2,y2, color = 'blue')
	# plt.scatter(x3,y3, color = 'green')

	# delta = 0.025
	# x = np.arange(-3, 3, delta)
	# y = np.arange(-3.5, 3.5, delta)
	# X, Y = np.meshgrid(x, y)
	# Z1 = mlab.bivariate_normal(X, Y, 1, 1, params[0][0][0], params[0][0][1])
	# Z2 = mlab.bivariate_normal(X, Y, 1, 1, params[1][0][0], params[1][0][1])
	# Z3 = mlab.bivariate_normal(X, Y, 1, 1, params[2][0][0], params[2][0][1])

	# CS = plt.contour(X, Y, Z1)
	# CS = plt.contour(X, Y, Z2)
	# CS = plt.contour(X, Y, Z3)
	# plt.clabel(CS, inline=1, fontsize = 10)
	# plt.title('Gaussians Obtained from Algorithm')

	# plt.show()


print "average success rate, 100 trials: " + str(np.mean(success_rates))
print "success rate standard deviation, 100 trials: " + str(np.std(success_rates))

print "average kmeans success rate, 100 trials: " + str(np.mean(kmeans_success_rates))
print "kmeans success rate standard deviation, 100 trials: " + str(np.std(kmeans_success_rates))

# startp = [([0,0], symmetriccov), ([0,0], symmetriccov), ([0,0], symmetriccov)]
# success_rates = []
# for i in range(0,100):
# 	print i
# 	datapoints = []

# 	while len(datapoints) < trials: #not exactly the trial number but whatever
# 		samples0 = np.random.multivariate_normal([0.95,0], [[0.5, 0],[0, 1]], s)
# 		samples120 = np.random.multivariate_normal(point120, cov120, s)
# 		samples240 = np.random.multivariate_normal(point240, cov240, s)

# 		for sample in samples0:
# 			datapoints.append((sample[0],sample[1],0))

# 		for sample in samples120:
# 			datapoints.append((sample[0],sample[1],120))

# 		for sample in samples240:
# 			datapoints.append((sample[0],sample[1],240))

# 	params = gradientDescent(startp, datapoints, 3)
# 	success_rate = 1-((getNumErrorsGaussian(datapoints, params))/float(len(datapoints)))
# 	print "success_rate: " + str(success_rate)
# 	success_rates.append(success_rate)

# print "average success rate [0,0], 100 trials: " + str(np.mean(success_rates))
# print "success rate standard deviation [0,0], 100 trials: " + str(np.std(success_rates))

# 3 GAUSSIANS
# segment length 3, start randomized hill climbing from [1,0], [0,1], [-1,0] : average success rate 69.3% += 6.7%
# segment length 3, start randomized hill climbing from [0,0] x 3 : average success rate 69.9% += 7.4%

# segment length 3, pick best neighbor from [1,0], [0,1], [-1,0] : average success rate, 100 trials: 0.708888888889 += 3.7%
# segment length 3, pick best neighbor from [0,0] x 3 : average success rate [0,0], 100 trials: 0.700917874396 += 6.9%


# From [0,0] x 3:
# average success rate, 100 trials: 0.703768115942
# success rate standard deviation, 100 trials: 0.0696117664979
# average kmeans success rate, 100 trials: 0.656183574879
# kmeans success rate standard deviation, 100 trials: 0.0906670882899




# 2 GAUSSIANS

# randomized hill climbing
# initial params of [0,0], [0,2]
# optimal params of [1,0], [1,2]
# segment length 3

# average optimal success rate 2 gaussians, 100 trials: 0.839903381643
# optimal success rate standard deviation 2 gaussians, 100 trials: 0.0256450311662
# average success rate 2 gaussians, 100 trials: 0.838309178744
# success rate standard deviation 2 gaussians, 100 trials: 0.026648588355
