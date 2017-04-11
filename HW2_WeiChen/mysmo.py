import numpy as np
import operator
import timeit
import matplotlib.pyplot as plt


def neg_y_t_gradient_t():

    gradient_t=Q.dot(alpha)-p
    return np.multiply(-y,gradient_t)

def find_i(neg_y_l_gradient_l):
    i=0
    max_i=neg_y_l_gradient_l[i,]
    for l in range(1,N):
        #note that this dataset use y=1 or 3,
        #I_up.append(l)
        if (alpha[l,]<C and y[l,]==1) or (alpha[l,]>0 and y[l,]==-1):
            curr_i=neg_y_l_gradient_l[l,]
            if curr_i>max_i:
                i=l
                max_i=curr_i
    return i, max_i

def find_j(neg_y_l_gradient_l,i,neg_y_i_gradient_i):
    j=0
    min_j=neg_y_l_gradient_l[j,]
    #print neg_y_i_gradient_i
    for l in range(1,N):
        #    I_low=[]
        if (alpha[l,]<C and y[l,]==-1) or (alpha[l,]>0 and y[l,]==1):
            #print neg_y_t_gradient_t(l), (neg_y_t_gradient_t(l)>neg_y_i_gradient_i)
            if neg_y_l_gradient_l[l,]<neg_y_i_gradient_i:
                a_it,b_it=cal_a_b(neg_y_l_gradient_l,i,l)
                curr_j=np.asscalar((b_it**2)/(-a_it))
                if curr_j<min_j:
                    j=l
                    min_j=curr_j
    #print "min_j", min_j
    return j, a_it, b_it

def cal_a_b(neg_y_l_gradient_l,i,t):
    a_it=np.inner(X[i,:],X[i,:])+np.inner(X[t,:],X[t,:])-(2.)*np.inner(X[i,:],X[t,:])
    b_it= neg_y_l_gradient_l[i,]-neg_y_l_gradient_l[t,]
    return a_it, b_it

def cal_obj(alpha):
    Q=np.multiply(X.dot(X.T),y.dot(y.T))
    #print (alpha.T.dot(Q).dot(alpha))/(2.0)
    #print sum(alpha,0)
    obj= np.asscalar((alpha.T.dot(Q).dot(alpha))/(2.0)-sum(alpha,0))
    #print obj.ndim, obj.shape
    return obj


def decomp():
    x_axis=[]
    y_axis=[]
    prev_obj=cal_obj(alpha)
    k=1
    #select working set B={i,j}
    while True:
        neg_y_l_gradient_l=neg_y_t_gradient_t()

        i, neg_y_i_gradient_i=find_i(neg_y_l_gradient_l)
        j, a_ij, b_ij= find_j(neg_y_l_gradient_l,i, neg_y_i_gradient_i)
    #solve the subproblem, set it as the k+1 input
        alpha[i,]=alpha[i,]+y[i,]*b_ij/a_ij
        alpha[j,]=alpha[j,]-y[j,]*b_ij/a_ij
    #calculate the object function to check stop criteria
        obj= cal_obj(alpha)
        k+=1
        x_axis.append(k)
        y_axis.append(obj)
        #print prev_obj-obj
        if  k>1000:
            return x_axis,y_axis
        prev_obj=obj
        #print x_axis
def mySmoSVM(filename, numruns):
    mnist = np.loadtxt(filename,delimiter=',')
    global N, X, y, alpha, Q, C,p
    N = mnist.shape[0]
    X=mnist[:,1:]
    y=mnist[:,0].reshape((N,1))
    y[y==1,]=-1
    y[y==3,]=1
    #initialize alpha
    alpha=np.zeros((N,1))
    C=0.1
    Q= np.multiply((y.dot(y.T)),(X.dot(X.T)))
    p=np.ones((N,1))
    time_list=np.zeros((numruns,))
    for q in range(numruns):
        start = timeit.default_timer()
        decomp()
        stop = timeit.default_timer()
        time_list[q,]=stop - start
    print "average time=",np.mean(time_list),"std=", np.std(time_list)
'''
    colours=['r','g','b','k','m']
    for q in range(5):
        x_axis, y_axis=decomp()
        plt.plot(x_axis, y_axis, colours[q])
    plt.xlabel('#iteration')
    plt.ylabel('obj')
    plt.title('dual_compare_5runs')
    plt.show()
'''
mySmoSVM('/Users/macbook/Documents/CSCI5525/MNIST-13.csv', 5)
