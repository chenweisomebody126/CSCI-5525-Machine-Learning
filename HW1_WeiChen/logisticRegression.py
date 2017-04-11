import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


def update_delta_w_multiclass(N,K,D,X,w,r):
    delta_w=np.zeros((K,D))
    for t in range(N):
        o_i=np.zeros((K,))
        for i in range(K):
            for j in range(D):
                o_i[i,]=o_i[i,]+w[i,j]*X[t,j]
        y_i=np.exp(o_i)
        y_i=y_i/float(np.sum(y_i))
        for i in range(K):
            for j in range(D):
                delta_w[i,j]=delta_w[i,j]+(r[t,i]-y_i[i,])*X[t,j]
    return delta_w

def update_w(w,ita,delta_w):
    w=w+ita*delta_w
    return w
def error_rate(w,file):
    if w.ndim>1:
        return error_rate_multiclass(w,file)
    else:
        return error_rate_twoclass(w,file)

def error_rate_twoclass(w,file):
    data=file[:,:-1]
    X=np.hstack((np.ones((data.shape[0],1)),data))
    target=file[:,-1]
    y=np.zeros((target.shape[0],))
    o=X.dot(w)
    o_exp=np.exp(-o)
    #print o_exp.shape
    for t in range(target.shape[0]):
        y_t=(1.)/(1+o_exp[t,])
        y[t,]=1 if y_t<0.5 else 0
    #print y_t
    error=y[y!=target].shape[0]/float(y.shape[0])
    return error

def error_rate_multiclass(w,file):
    data=file[:,:-1]
    X=np.hstack((np.ones((data.shape[0],1)),data))
    target=file[:,-1]
    y=np.zeros((target.shape[0],))
    o=X.dot(w.T)
    #print o.shape,o
    o_exp=np.exp(o)
    #print o_exp.shape
    for t in range(target.shape[0]):
        sum_all_o=np.sum(o_exp[t,:])
        #print o_exp[t,:],sum_all_o
        y_t=o_exp[t,:]/float(sum_all_o)
        y[t,]=np.argmax(y_t)
    #print y_t
    error=y[y!=target].shape[0]/float(y.shape[0])
    return error

def trainLR(file):
    target=file[:,-1]
    K=np.unique(target).shape[0]
    if K<=2:
        return trainLR_twoclass(file)
    else:
        return trainLR_multiclass(file)

def trainLR_multiclass(file):
    target=file[:,-1]
    K=np.unique(target).shape[0]
    data=file[:,:-1]
    N=data.shape[0]
    X=np.hstack((np.ones((N,1)),data))
    #print target
    #generate label matrix N*K, full of 0 and 1
    r=np.zeros((N,K))
    for t in range(N):
        tag=target[t,]
        r[t,int(tag)]=1
    #print r
    D=X.shape[1]
    w=np.random.uniform(-0.01,0.01,(K,D))
    #print w
    #repeat until converge:0.002>difference>0
    ita=0.0001

    new_error=0
    difference=1
    iteration=0
    while difference>0.002 or difference<0 or iteration<50:
        delta_w=update_delta_w_multiclass(N,K,D,X,w,r)
        w=update_w(w,ita,delta_w)
        previous_error=new_error
        new_error= error_rate_multiclass(w,file)
        iteration+=1
        difference=previous_error-new_error
    return w
def trainLR_twoclass(file):
    r=file[:,-1]
    data=file[:,:-1]
    N=data.shape[0]
    X=np.hstack((np.ones((N,1)),data))
    D=X.shape[1]
    w=np.random.uniform(-0.01,0.01,(D,))

    ita=0.0001
    new_error=0
    difference=1
    iteration=0
    #repeat until converge
    while difference>0.002 or difference<0 or iteration<50:
            delta_w=update_delta_w_twoclass(N,D,X,w,r)
            w=update_w(w,ita,delta_w)
            previous_error=new_error
            new_error= error_rate_twoclass(w,file)
            iteration+=1
            difference=previous_error-new_error
    return w
def update_delta_w_twoclass(N,D,X,w,r):
    delta_w=np.zeros((D,))
    for t in range(N):
        o=0
        for j in range(D):
            o=o+w[j,]*X[t,j]
        y=1/(1+np.exp(-o))
        for j in range(D):
            delta_w=delta_w+(r[t,]-y)*X[t,j]
    return delta_w


def train_test_split(file, train_percent):
    N=file.shape[0]
    test_size=N/5
    test_start=np.random.randint(0,N-test_size)
    test_file=file[test_start:test_start+test_size,:]
    train_file=np.vstack((file[:test_start,:],file[test_start+test_size:,:]))
    #generate train_file for each train percent
    error_split=np.zeros((len(train_percent),))
    for i, percent in enumerate(train_percent):
        train_end=(train_file.shape[0]*percent)/100

        train_percent_file=train_file[:train_end,:]

        w=trainLR(train_percent_file)

        error_split[i,]=error_rate(w,test_file)

    return error_split

def learning_curve(train_percent,error_mean_percent,error_std):
    x=train_percent
    y=error_mean_percent
    plt.figure()
    plt.errorbar(x, y, yerr=error_std)
    plt.show()
    return

def logisticRegression(filename,num_splits,train_percent):
    file = genfromtxt(filename, delimiter=',')
    error_matrix=np.zeros((num_splits,len(train_percent)))
    for i in range(num_splits):
        error_matrix[i,:]=train_test_split(file,train_percent)
    error_mean_percent=np.mean(error_matrix,axis=0)
    error_std=np.zeros((len(train_percent),))
    for j in range(len(train_percent)):
        error_std[j,]=np.std(error_matrix[:,j])
    print error_mean_percent
    print error_matrix
    print error_std
    learning_curve(train_percent,error_mean_percent,error_std)
    return

logisticRegression('/Users/macbook/Downloads/digits.csv', 10,[10,25,50,75,100])
