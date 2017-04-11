import numpy as np

def projectVector(file):
    data=file[:,:-1]
    #print data
    target=file[:,-1]
# withinClassMatrix: S_w
    d=data.shape[1] # number of dimensions
    #mean for each class
    meanVectors=[]
    for i in range(10):
        meanVectors.append(np.mean(data[target==i],axis=0))
    #print meanVectors
    S_w=np.zeros((d,d))
    for i in range(len(meanVectors)):
        S_i=np.zeros((d,d))
        for row in data[target==i]:
            S_i+=np.outer((row-meanVectors[i]),(row-meanVectors[i]))
        S_w+=S_i
# betweenClassMatrix: S_b
    entireMean=np.mean(data)
    S_b=np.zeros((d,d))
    for i, iMean in enumerate(meanVectors):
        n_i=data[target==i].shape[0]
        S_b+=n_i*np.outer(iMean-entireMean,iMean-entireMean)
#calculate the largest eigenvectors for inv(S_w)*S_b
    eigVals, eigVecs= np.linalg.eig(np.linalg.pinv(S_w).dot(S_b))
    eig_pairs=[(np.abs(eigVals[i,]), eigVecs[i,]) for i in range(eigVals.shape[0])]
    eig_pairs.sort(key=lambda x:x[0],reverse=True )
    projectMatrix=eigVecs[:2,]
    return projectMatrix

def generative(X,target):
    Sigma_all=np.zeros((20,2))
    mean_all=np.zeros((10,2))
    prior_all=np.zeros((10,))
    for k in range(10):
        X_k=X[target==k]
        mean_k=np.mean(X_k,axis=0)
        mean_all[k,:]=mean_k
        Sigma_k=np.zeros((2,2))
        for row in X_k:
            Sigma_k=np.add(Sigma_k,np.outer((row-mean_k),(row-mean_k)))
        Sigma_all[2*k:2*(k+1),:]=Sigma_k/float(X_k.shape[0])
        prior_all[k,]=X_k.shape[0]
    return prior_all,mean_all,Sigma_all


#cross validation
def cross_validation(file,num_crossval,projectMatrix):
    #train_test_split
    X=file[:,:-1]
    target=file[:,-1]
    X=np.dot(X,projectMatrix.T).real
    n=target.shape[0]
    fold_size=n/num_crossval
    error_train=np.zeros((num_crossval,))
    error_test=np.zeros((num_crossval,))
    for i in range(num_crossval):
        if i<num_crossval-1:
            X_test=X[i*fold_size:(i+1)*fold_size]
            X_train=np.vstack((X[:i*fold_size,:],X[(i+1)*fold_size:,:]))
            target_test=target[i*fold_size:(i+1)*fold_size]
            target_train=np.hstack((target[:i*fold_size,],target[(i+1)*fold_size:,]))
        else:
            X_test=X[i*fold_size:]
            X_train=X[:i*fold_size]
            target_test=target[i*fold_size:]
            target_train=target[:i*fold_size]

        prior_all, mean_all, Sigma_all=generative(X_train,target_train)
        error_train[i,]=error_rate(X_train,target_train,prior_all, mean_all, Sigma_all)
        error_test[i,]=error_rate(X_test, target_test, prior_all, mean_all, Sigma_all)

    print "train error rate: ",np.sum(error_train)/float(num_crossval)
    print "test error rate:", np.sum(error_test)/float(num_crossval)
    print "test error standard deviation:",  np.std(error_test)

def error_rate(X,target, prior_all, mean_all, Sigma_all):
    N=X.shape[0]
    y=np.zeros((N,))
    for i in range(N):
        poster=np.zeros((10,))
        for j in range(10):
            b=np.log(np.linalg.norm(Sigma_all[j*2:(j+1)*2,:]))/(-2.)
            a=(X[i,]-mean_all[j,])
            aT=(X[i,]-mean_all[j,]).reshape(1,2)
            #print Sigma_all[j*2:(j+1)*2,:]
            Sigma_inv=np.linalg.pinv(Sigma_all[j*2:(j+1)*2,:])
            poster[j,]=b+aT.dot(Sigma_inv).dot(a)/(-2.)+np.log(prior_all[j,])
        index=np.argmax(poster)
        y[i,]=index
    error=y[y!=target].shape[0]/float(N)
    return error



def LDA2dGaussGM(filename, num_crossval):
    from numpy import genfromtxt
    file = genfromtxt(filename, delimiter=',')
    projectMatrix=projectVector(file)
    cross_validation(file,num_crossval,projectMatrix)
    return

LDA2dGaussGM('/Users/macbook/Downloads/boston50.csv', 10)
