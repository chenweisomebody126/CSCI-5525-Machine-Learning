import numpy as np
import matplotlib.pyplot as plt


def trainNB(file):
    X=file[:,:-1]
    target=file[:,-1]
    K=np.unique(target).shape[0]
    D=X.shape[1]

    Sigma_all=np.zeros((K,D))
    mean_all=np.zeros((K,D))
    prior_all=np.zeros((K,))

    for k in range(K):
        X_k=X[target==k]
        mean_all[k,:]=np.mean(X_k,axis=0)
        for j in range(D):
            Sigma_all[k,j]=np.var(X_k[:,j])
        prior_all[k,]=X_k.shape[0]
    return prior_all,mean_all,Sigma_all

def error_rate(file, prior_all, mean_all, Sigma_all):
    X=file[:,:-1]
    target=file[:,-1]
    K=Sigma_all.shape[0]
    N=X.shape[0]
    D=Sigma_all.shape[1]
    y=np.zeros((N,))
    for i in range(N):
        poster=np.zeros((K,))
        for k in range(K):
            b=0
            for j in range(D):
                if Sigma_all[k,j]==0:
                    Sigma_all[k,j]=0.000001
                b=b+((X[i,j]-mean_all[k,j])**2)/((-2.)*Sigma_all[k,j])
            a=np.sum(np.log(Sigma_all[k,:]))/(-2.)
            poster[k,]=a+b+np.log(prior_all[k,])
        index=np.argmax(poster)
        y[i,]=index
    error=y[y!=target].shape[0]/float(N)
    return error

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
        #split percent of train set
        train_percent_file=train_file[:train_end,:]
        #estimate mean and sigma
        prior_all,mean_all,Sigma_all=trainNB(train_percent_file)
        #use mean, sigma to classfy test data and calculate error_rate
        error_split[i,]=error_rate(test_file, prior_all, mean_all, Sigma_all)
    return error_split
#plot learning_curve
def learning_curve(train_percent,error_mean_percent,error_std):
    x=train_percent
    y=error_mean_percent
    plt.figure()
    plt.errorbar(x, y, yerr=error_std)
    plt.show()
    return 


def naiveBayesGaussion(filename,num_splits,train_percent):
    from numpy import genfromtxt
    file = genfromtxt(filename, delimiter=',')
    error_matrix=np.zeros((num_splits,len(train_percent)))
    for i in range(num_splits):
        error_matrix[i,:]=train_test_split(file,train_percent)
    error_mean_percent=np.mean(error_matrix,axis=0)
    error_std=np.zeros((len(train_percent),))
    for j in range(len(train_percent)):
        error_std[j,]=np.std(error_matrix[:,j])
    print "error_mean_for_each_percent:",error_mean_percent
    print "error_matrix_for (num_split,train_percent):",error_matrix
    print "error_rate_standard_deviation: ",error_std
    learning_curve(train_percent,error_mean_percent,error_std)
    return


naiveBayesGaussion('/Users/macbook/Downloads/digits.csv', 10,[10,25,50,75,100])
