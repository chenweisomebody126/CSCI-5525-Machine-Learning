import numpy as np
import random
import timeit
import matplotlib.pyplot as plt

def cal_gradient_i(w, i):
    g=lmda*w
    if y[i,]*np.inner(w, X[i,:])<1:
        g=g-y[i,]*X[i,:]
    return g
def cal_obj(w):
    loss_sum=0
    for i in range(N):
        loss= max(0, 1-y[i,]*(w.T.dot(X[i,:])))
        loss_sum+=loss
    obj= lmda/(2.)*w.T.dot(w)+loss_sum/float(N)
    #print "w**2",lmda/(2.)*w.T.dot(w),"loss", loss_sum/float(N)
    return np.asscalar(obj)


def svrg(m):
    w_s=np.zeros((d,))
    ita=0.0001
    #initialize update frequency m, learning rate ita, w
    k_tot=1
    obj_vec= np.zeros((100,))
    outer_iter=1
    x_axis=[]
    y_axis=[]

    while True:
        w=w_s
        #iteration over n gradient descent
        g=np.zeros((d,))
        for i in range(N):
            if k_tot>100*N:
                print "obj=",cal_obj(w_m),"outer_iter=", outer_iter
                return x_axis, y_axis
            #if k_tot%N==0: obj_vec[k_tot/N]=cal_obj(w)
            #if k_tot%N==0: print cal_obj(w)
            #print cal_obj(w)
            g+=cal_gradient_i(w, i)
            k_tot+=1
        mu=g/float(N)
        #iteration over m stochoastic gradient descent
        w_m=w
        for t in range(m):
            if k_tot>100*N:
                print "obj=",cal_obj(w_m),"outer_iter=", outer_iter
                return x_axis, y_axis
            #if k_tot%N==0: obj_vec[k_tot/N]=cal_obj(w)
            #if k_tot%N==0: print cal_obj(w)
            i_t=random.randint(0,N-1)
            w_m=w_m-ita*(cal_gradient_i(w_m,i_t)-cal_gradient_i(w,i_t)+mu)
            k_tot+=1
            x_axis.append(k_tot)
            y_axis.append(cal_obj(w_m))
        w_s=w_m
        outer_iter+=1
        if outer_iter>1000:
            print "obj=",cal_obj(w_m),"outer_iter=", outer_iter
            return x_axis, y_axis

def mySVRG(filename, m,numruns):
    mnist = np.loadtxt(filename,delimiter=',')
    global N,d, X, y, lmda
    N = mnist.shape[0]
    X=mnist[:,1:]
    d = X.shape[1]
    y=mnist[:,0].reshape((N,1))
    y[y==1,]=-1
    y[y==3,]=1
    lmda=0.0001

    for q in range(numruns):
        start = timeit.default_timer()
        svrg(m)
        stop = timeit.default_timer()
        time_list[q,]=stop - start
    print "average time=",np.mean(time_list),"std=", np.std(time_list)

    '''
    colours=['r','g','b','k','m']
    for run in range(5):
        x_axis, y_axis=svrg(200)
        plt.plot(x_axis, y_axis, colours[run])
    plt.xlabel('#grad')
    plt.ylabel('obj')
    plt.show()
    '''


mySVRG('/Users/macbook/Documents/CSCI5525/MNIST-13.csv',20,5)
