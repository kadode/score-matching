# -*- coding:utf-8 -*-
import numpy as np
import argparse
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=2, help="")
parser.add_argument('--num_sample', type=int, default=30, help="")
parser.add_argument('--epoch', type = int, default=10, help ="if cuda is available, program will use cuda.")
parser.add_argument('--lr', type = float, default=0.1, help ="")
parser.add_argument('--batch_size', type = int, default=1, help ="")
args = parser.parse_args()

mean = [10, 5]
#cov = [[1, 0], [0, 1]]  # diagonal covariance
cov = [[3, 5], [5, 1]]  # diagonal covariance

x = np.random.multivariate_normal(mean, cov, args.num_sample).T
x = np.asmatrix(x)
#print(x*x.T)
#print(x[:,0])
#print(np.sum(x,axis=1)/args.num_sample)
#print(np.sum(x,axis=1))
'''
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
'''
I = np.matrix(np.eye(args.dim))
M = np.matrix(np.random.uniform(0,1,(args.dim,args.dim)))
mu = np.matrix(np.random.random(args.dim)).T
#print(M)


J = 0.0
for i in range(args.num_sample):
    a = x[:,i]-mu
    b = M * a
    J += b.T*b/2
    for j in range(args.dim):
        J -= M[j,j]
J = J/args.num_sample
print("Objective Function : {} | f norm : {}".format(J,np.linalg.norm(M-np.matrix(cov))))
best_J = J
lr = args.lr
num_batches = args.num_sample // args.batch_size + args.num_sample%args.batch_size
remainder = args.num_sample - num_batches * args.batch_size
batches = [x[:,j*args.batch_size:(j+1)*args.batch_size] for j in range(num_batches-1)]
batches.append(x[:,(num_batches-1)*args.batch_size:])
stop_count = 0
for k in range(args.epoch):
    for j in range(num_batches):
        s = batches[j].shape[1]
        grad_mu = M*M*(mu-(np.sum(batches[j],axis=1)/s))
        A = np.matrix(np.zeros((args.dim,args.dim)))
        for i in range(s):
            a = batches[j][:,i] - mu
            A += a*a.T
        A = A / (2*s)
        grad_M = - I + M*A + A*M

        M -= grad_M * lr
        mu -= grad_mu * lr

    J = 0.0
    for i in range(args.num_sample):
        a = x[:,i]-mu
        b = M * a
        J += b.T*b/2
        for j in range(args.dim):
            J -= M[j,j]
    J = J/args.num_sample
    print("Epoch {} | Loss {} | best loss {} | f norm {} | stop count {}".format(
        k+1,J,best_J,np.linalg.norm(mean-mu)+np.linalg.norm(M-np.matrix(cov)),stop_count))
    if stop_count > 4:
        break
    if J > best_J:
        stop_count += 1
        lr = lr/4
    else:
        best_J = J
        stop_count = 0
print("True parameter")
print(mean)
print(cov)
print("Estimated parameter")
print(mu)
print(M)
