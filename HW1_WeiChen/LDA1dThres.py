import numpy as np

def projectVector(data, target):
# withinClassMatrix: S_w
    d=data.shape[1] # number of dimensions

    #mean for each class
    meanVectors=[]
    for i in [0,1]:
        meanVectors.append(np.mean(data[target==i],axis=0))
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
    eig_pairs=[(np.abs(eigVals[i]), eigVecs[i]) for i in range(len(eigVals))]
    eig_pairs.sort(key=lambda x:x[0],reverse=True )
#calculate projection in new 1d space
    threshold=(meanVectors[0]+meanVectors[1])/2
    threshold_project=threshold.dot(eig_pairs[0][1])
    return (eig_pairs[0][1], threshold_project)

def error_rate(X,target,w,threshold_project):
    X_project=np.dot(X,w)
    y_project=np.zeros(target.shape[0],)
    y_project[X_project<threshold_project]=1
    error=float(y_project[y_project!=target].shape[0])/y_project.shape[0]
    return error


#cross validation
def cross_validation(X,target,num_crossval,w,threshold_project):
    #train_test_split
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
        #classify in new subspace and calculate error_rate
        error_train[i,]=error_rate(X_train,target_train,w,threshold_project)
        error_test[i,]=error_rate(X_test, target_test, w, threshold_project)
    print "train error rate: ",np.sum(error_train)/float(num_crossval)
    print "test error rate:", np.sum(error_test)/float(num_crossval)
    print "test error standard deviation:",  np.std(error_test)
    return

def LDA1dThres(filename,num_crossval):
    from numpy import genfromtxt
    file = genfromtxt(filename, delimiter=',')
    X=file[:,:-1]
    target=file[:,-1]
    (w,thre)=projectVector(X,target)
    cross_validation(X,target,num_crossval,w,thre)
    return

LDA1dThres('/Users/macbook/Downloads/boston50.csv',10)
