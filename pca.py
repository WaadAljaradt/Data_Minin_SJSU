
import numpy as np
import numpy.linalg as LA

#read data from data.txt 
#the data is from the UCR Wine dataset, found at https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data (Links to an external site.)
#
array = np.loadtxt('data.txt',delimiter=',', usecols=range(0,14))

# remove the first column in the data which are class labels
data= array[:,1:]


# first : standardize the data by centering the variables :
data -= np.mean(data, axis=0)

# Second : calculate the covariance matrix 
C = np.corrcoef(data, rowvar=0)


# Third : get the eigenvalues and the eigenvectors of the covariance matrix:
eval, evec = LA.eig(C)

#print eigenvalues of the matrix 
print(eval)

#Fourth normalize the eigenvalues and sort it by decreasing order
tot = sum(eval)
var_exp = [(i / tot)*100 for i in sorted(eval, reverse=True)]

#Fifth calculate the accumulated variance percentage contribution for each principle component 
#all must add to 100
cum_var_exp = np.cumsum(var_exp)

#print principle component variance explained by each feature 
print(var_exp)
#print principle component variability variance explained in an accumulated percentage for each feature 
print(cum_var_exp)






