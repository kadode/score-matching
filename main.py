# -*- coding:utf-8 -*-
import os
import inspect
import numpy as np
import argparse
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=2, help="")
parser.add_argument('--num_sample', type=int, default=30, help="")
parser.add_argument('--epoch', type = int, default=10, help ="")
parser.add_argument('--lr', type = float, default=0.1, help ="")
parser.add_argument('--batch_size', type = int, default=1, help ="")
args = parser.parse_args()

#mean = [10, 5]
mean = np.random.randint(-10,10,args.dim)
#mean = np.random.random(args.dim)

if not os.path.isdir("fig"):
    os.makedirs("fig")

print(mean)
#cov = [[1, 0], [0, 1]]  # diagonal covariance
#cov = [[3, 5], [5, 1]]  # diagonal covariance
#cov = np.matrix(np.eye(args.dim)) * 5.0
#x = np.matrix(np.random.rand(args.dim,args.dim))
x = np.matrix([[4,-5],[1,7]])
cov = x*x.T

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
losses = []
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
    losses.append(J[0,0])
    invM = np.linalg.inv(M)
    print("Epoch {} | Loss {} | best loss {} | f norm {} | stop count {}".format(
        k+1,J[0,0],best_J[0,0],np.linalg.norm(mean-mu)+np.linalg.norm(invM-np.matrix(cov)),stop_count))
    if stop_count > 4:
        break
    if J > best_J:
        stop_count += 1
        lr = lr/4
    else:
        best_J = J
        stop_count = 0
        
    if args.dim == 2:
        w = 4
        gx,gy = np.mgrid[mean[0]-w:mean[0]+w:.1,mean[1]-w:mean[1]+w:.1]
        pos = np.dstack((gx,gy))
        rv = multivariate_normal(mean,cov)
        rv2 = multivariate_normal([mu[0,0],mu[1,0]],invM.tolist())
        plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(2,2)

        plt.subplot(gs[0,0])
        plt.contourf(gx,gy,rv.pdf(pos))
        
        plt.subplot(gs[0,1])
        plt.contourf(gx,gy,rv2.pdf(pos))
        
        plt.subplot(gs[1,:])
        #plt.set_xlim(0.0,args.epoch+1)
        ind = [j+1 for j in range(k+1)]
        plt.xlim(0,args.epoch+1)
        #plt.ylim(-0.3,0.0)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(ind,losses,'x')
        '''
        ax = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax.contourf(gx,gy,rv.pdf(pos))
        ax2.contourf(gx,gy,rv2.pdf(pos))
        ind = [j+1 for j in range(k+1)]
        print(ind)
        print(losses)
        ax3.set_xlim(0.0,args.epoch+1)
        ax3.plot(ind,losses)
        '''
        '''
        for x in inspect.getmembers(ax3,inspect.ismethod):
            print(x[0])
        '''
        #plt.show()
        plt.savefig(os.path.join('fig',"{}".format(k).zfill(8)))
        plt.close("all")

print("True parameter")
print(mean)
print(cov)
print("Estimated parameter")
print(mu)
print(np.linalg.inv(M))
