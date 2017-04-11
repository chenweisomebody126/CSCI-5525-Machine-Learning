import numpy as np
from numpy import linalg
import random
import timeit
import matplotlib.pyplot as plt

def cal_obj(w):
    loss_sum=0
    for i in range(N):
        loss= max(0, 1-y[i,]*(w.T.dot(X[i,:])))
        loss_sum+=loss
    obj= lmda/(2.)*w.T.dot(w)+loss_sum/float(N)
    #print "w**2",lmda/(2.)*w.T.dot(w),"loss", loss_sum/float(N)
    return np.asscalar(obj)

def mini_batch(k):
    w=np.zeros((d,1))
    outer_iter=1
    k_tot=0

    x_axis=[]
    y_axis=[]

    while True:
        #randomly choose k records
        A_t=random.sample(xrange(N), k)
        #find the records y*f(x)<0 in A_t
        #set step size
        ita=(1.)/(lmda*outer_iter)
        sum_y_x=np.zeros((d,1))
        #A_t_plus=[]
        for i in A_t:
            if y[i,]*w.T.dot(X[i,:])<1:
                #A_t_plus.append(i)
                sum_y_x+=y[i,]*(X[i,:][:,None])
                k_tot+=1

                if k_tot>100*N:
                    print "obj=",cal_obj(w),"outer_iter=", outer_iter
                    return x_axis, y_axis
        #calculate w_t+1/2, denoted by w_half
        w_half= ((1.)-ita*lmda)*w+ita/k*sum_y_x
        #update w
        min_one=min(1, 1./(np.sqrt(lmda)*np.linalg.norm(w_half)) )
        w=min_one*w_half
        #obj=cal_obj()
        #print obj
        x_axis.append(k_tot)
        y_axis.append(cal_obj(w))
        outer_iter+=1
        if outer_iter>1000:
            print "obj=",cal_obj(w),"outer_iter=", outer_iter
            return x_axis, y_axis
        #print outer_iter
        #print cal_obj(w)

def myPegasos(filename, k, numruns):
    mnist = np.loadtxt(filename,delimiter=',')
    global N,d, X, y,lmda
    N = mnist.shape[0]
    X=mnist[:,1:]
    d = X.shape[1]
    y=mnist[:,0].reshape((N,1))
    y[y==1,]=-1
    y[y==3,]=1
    lmda=9

    time_list=np.zeros((numruns,))
    for q in range(numruns):
        start = timeit.default_timer()
        mini_batch(k)
        stop = timeit.default_timer()
        time_list[q,]=stop - start
    print "average time=",np.mean(time_list),"std=", np.std(time_list)

'''
colours=['r','g','b','k','m']
for run in range(5):
    x_axis, y_axis=mini_batch(200)
    plt.plot(x_axis, y_axis, colours[run])
plt.xlabel('#grad')
plt.ylabel('obj')
plt.title('pegasos200')
plt.show()
'''
myPegasos('/Users/macbook/Documents/CSCI5525/MNIST-13.csv',20,5)
