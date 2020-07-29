import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer

from sklearn.decomposition import PCA
filename='./Q1.txt'
file= open(filename)
lists = []
for line in file.readlines():
    nline = line.strip().split(' ')
    new_list=[x for x in nline if x!='']
    print(new_list)
    lists.append(new_list)

head=lists[0]
del head[0]
del lists[0] # delete the first line of variable names

dataset=[]
for line in lists:
    line.pop(0)
    num_i = [float(i) for i in line]
    dataset.append(num_i)

array_data = np.array(dataset)

corr_matrix = np.corrcoef(array_data.T)
# np.fill_diagonal(corr_matrix, 0)
new_corr_matrix = corr_matrix- np.diag(np.diag(corr_matrix))
pos = np.unravel_index(new_corr_matrix.argmax(), new_corr_matrix.shape)
val = new_corr_matrix.max()
# Return a tuple of (3,6) and a value of 0.6435501422986784
NameX=head[pos[0]]
NameY=head[pos[1]]

# Return 'Nonwhite' and  'Mortality'.
X=array_data[:,pos[0]]
Y=array_data[:,pos[1]]
r, p= stats.pearsonr(X, Y)

# p=2.922261492295322e-08

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
plt.plot(X, Y, 'o', label='original data')
plt.plot(X, intercept + slope*X, 'r', label='fitted line')
plt.legend()
plt.show()

print("r-squared: %f" % r_value**2)
# r-squared: 0.414157

fa = FactorAnalyzer()
fa.fit(array_data)
ev, cfev=fa.get_eigenvalues()
# there are 3 eigenvalues  greater than one considered as the number of factors
print("The loading matrix is:\n {}".format(fa.loadings_))

print("The uniquenesses is :\n{}".format(fa.get_uniquenesses()))

"""
(1)
    (a)最相关：'Nonwhite' and  'Mortality' ，相关系数为：0.6435501422986784。
	(b) 非常显著。p值为：p=2.922261492295322e-08远小于0.01
	(c) X和Y之间线性关系不显著。确定性系数R方=0.414157
	
(2)
	(a) 根据EFA结果。有3个因子比较合适。根据Kaiser 标准，利用本征值决定因子个数，大于1，则选择该因子。
	(b) array([[-0.76959259,  0.20848783, -0.11848051],
       [ 0.57634158,  0.04729659, -0.38117515],
       [-0.02239135, -0.14389588,  0.58653842],
       [ 0.05305706,  1.0883023 , -0.23383744],
       [ 0.70635283,  0.16207034,  0.34302476],
       [ 0.24004869,  0.07169364,  0.82317674],
       [-0.33801322,  0.47714946,  0.36514599]])
	(c) 假设三个因子分别为F1，F2，F3。 F1因子主要由'Rainfall'和'NOX'所决定，F2因子主要由'Nonwhite'所决定，
	F3因子主要由 'SO2'和'Popden'所决定
	(d)array([ 0.35022245,  0.52029892,  0.63476528, -0.2418969 ,  0.35713291,
        0.2596167 ,  0.52474386])
	uniqueness代表了误差和未能被因子所解释的那部分。大部分变量的uniqueness值都很小，小于0.6，代表了这些选择的因子很好的解释了原始变量。
"""
