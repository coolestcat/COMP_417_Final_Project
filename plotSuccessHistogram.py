import numpy as np
import matplotlib.pyplot as plt

files = ['three_sparse_uniform_3D_dists.txt', 'three_sparse_uniform_dists.txt', 'three_unequal_uniform_dists.txt', '2_overlapping_uniform_dists.txt', 'linear_dists.txt', 'gaussian_dists.txt']
titles = ['Three Uniform Distributions', 'Three 3D Uniform Distributions', 'Three Unequal Uniform Distributions', 'Two Overlapping Uniform Distributions', 'Linear Classifier', 'Gaussian Distributions']

i = 0
for f in files:

	thisFile = open(f, 'r')

	success_rates = []
	for line in thisFile:
		l = line.rstrip()
		if l[0:13] == 'success_rate:' or l[0:13] == 'success rate:':
			rate = float(l[14:len(l)])
			success_rates.append(rate)

	plt.figure()
	plt.hist(success_rates, bins=20)
	plt.ylim(0,40)
	plt.xlabel('Success Rate %')
	plt.ylabel('Runs')
	plt.title(titles[i])
	i += 1
	plt.show()

	thisFile.close()