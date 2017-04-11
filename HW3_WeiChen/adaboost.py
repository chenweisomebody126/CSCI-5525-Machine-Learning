import numpy as np

def readCsvToBinary(filename):
    file = np.genfromtxt(filename, delimiter=',')
    X=file[:,:-1]
    target=file[:,-1]
    tao50= np.median(target)
    tao75=np.percentile(target,75)
    target50=np.copy(target)
    target75=np.copy(target)
    target50[target50>=tao50]=1
    target50[target50!=1]=-1
    target75[target75>=tao75]=1
    target75[target75!=1]=-1
    print "tao",tao50, tao75
    return X, target50,target75

def boosting(X_target,b):
    N=X_target.shape[0]
    w_t=np.ones((N,))*(1./N)
    boosted=[]
    for t in range(b):
        X_target_w_t=np.hstack((X_target,w_t[:,None]))
        feature_split_tree,leaf_class=generateWeakLearner(X_target_w_t)
        alpha_t,w_t=Update_w_t(X_target_w_t,feature_split_tree,leaf_class)
        weak_learner_t=(feature_split_tree,leaf_class,alpha_t)
        #print "weaklearner", weak_learner_t
        boosted.append(weak_learner_t)
    return boosted

def Update_w_t(X_target_w_t,feature_split_tree,leaf_class):
    error_w_t=0.
    N=X_target_w_t.shape[0]
    class_t=np.zeros((N,))
    for i in range(N):
        if X_target_w_t[i,feature_split_tree[0][0]]<feature_split_tree[0][1]:
            if X_target_w_t[i,feature_split_tree[1][0]]<feature_split_tree[1][1]:
                class_t[i,]=leaf_class[0]
            else:
                class_t[i,]=leaf_class[1]
        else:
            if X_target_w_t[i,feature_split_tree[2][0]]<feature_split_tree[2][1]:
                class_t[i,]=leaf_class[2]
            else:
                class_t[i,]=leaf_class[3]
        if class_t[i,]!=X_target_w_t[i,-2]:
            error_w_t+=X_target_w_t[i,-1]
    e_t=float(error_w_t)/np.sum(X_target_w_t[:,-1])
    #print "e_t", e_t
    #print class_t, X_target_w_t[:,-2]
    alpha_t=0.5*np.log((1.-e_t)/e_t)
    w_t_new=X_target_w_t[:,-1]*np.exp(-alpha_t*X_target_w_t[:,-2]*class_t)
    w_t_new=w_t_new/np.sum(w_t_new)
    #print "alpha_t",alpha_t,"e_t",e_t
    return alpha_t,w_t_new

def classifyAndError(X_target,boosted):#alpha_t,feature_split_tree,leaf_class
    error_i=0.
    N=X_target.shape[0]
    class_i=np.zeros((N,))
    for i in range(N):
        class_i_t=0.
        for t, weak_learner_t in enumerate(boosted):
            feature_split_tree=weak_learner_t[0]
            leaf_class=weak_learner_t[1]
            alpha_t=weak_learner_t[2]

            if X_target[i,feature_split_tree[0][0]]<feature_split_tree[0][1]:
                if X_target[i,feature_split_tree[1][0]]<feature_split_tree[1][1]:
                    class_i_t=leaf_class[0]
                else:
                    class_i_t=leaf_class[1]
            else:
                if X_target[i,feature_split_tree[2][0]]<feature_split_tree[2][1]:
                    class_i_t=leaf_class[2]
                else:
                    class_i_t=leaf_class[3]

            class_i[i,]+=class_i_t*alpha_t

        if class_i[i,]>=0:
            class_i[i,]=1
        else:
            class_i[i,]=-1

        if class_i[i,]!=X_target[i,-1]:
            error_i+=1
    e_t=float(error_i)/N
    return e_t

def generateWeakLearner(X_target_w_t):
    feature_root,split_value_root=generateLayer(X_target_w_t)
    X_target_w_t_left_node=X_target_w_t[X_target_w_t[:,feature_root]<split_value_root,:]
    X_target_w_t_right_node=X_target_w_t[X_target_w_t[:,feature_root]>=split_value_root,:]

    feature_left,split_value_left= generateLayer(X_target_w_t_left_node)
    target_left_left=X_target_w_t_left_node[X_target_w_t_left_node[:,feature_left]<split_value_left,-2]
    target_left_right=X_target_w_t_left_node[X_target_w_t_left_node[:,feature_left]>=split_value_left,-2]
    left_left_class=decideLeaf(target_left_left)
    left_right_class=decideLeaf(target_left_right)

    feature_right,split_value_right= generateLayer(X_target_w_t_right_node)
    target_right_left=X_target_w_t_right_node[X_target_w_t_right_node[:,feature_right]<split_value_right,-2]
    target_right_right=X_target_w_t_right_node[X_target_w_t_right_node[:,feature_right]>=split_value_right,-2]
    right_left_class=decideLeaf(target_right_left)
    right_right_class=decideLeaf(target_right_right)

    feature_split_tree=[(feature_root,split_value_root),(feature_left,split_value_left),(feature_right,split_value_right)]
    leaf_class=[left_left_class,left_right_class,right_left_class,right_right_class]
    return feature_split_tree,leaf_class

def decideLeaf(target_leaf):
    count_pos_leaf=target_leaf[target_leaf==1].shape[0]
    count_neg_leaf=target_leaf[target_leaf==-1].shape[0]
    if count_neg_leaf<=count_pos_leaf:
        return 1
    else:
        return -1

def generateLayer(X_target_w_t_subtree):
    max_IG_feature=-np.inf
    for feature in range(13):
#for earch feature, each find the split with max informaiton gain within feature
#second, find the max information gain across all features
        IG_feature,feature_split_value=informationGain(X_target_w_t_subtree,feature)
        #print "feature",feature, "IG",IG_feature, "split",feature_split_value
        if IG_feature>max_IG_feature:
            max_IG_feature=IG_feature
            best_feature=feature
            best_feature_split_value=feature_split_value
    return best_feature,best_feature_split_value

def informationGain(X_target_w_t_subtree,feature):
    entropy_before=entropy(X_target_w_t_subtree[:,-2:])
    if feature==3:
        X_target_w_t_left_node=X_target_w_t_subtree[X_target_w_t_subtree[:,feature]==0]
        X_target_w_t_right_node=X_target_w_t_subtree[X_target_w_t_subtree[:,feature]==1]
        best_split_value=1
        best_split_entropy_after=conditionalEntropy(X_target_w_t_left_node[:,-2:],X_target_w_t_right_node[:,-2:])
    else:
        best_split_entropy_after=np.inf
        for split_quantile in range(10,100,10):
            split_value=np.percentile(X_target_w_t_subtree[:,feature],split_quantile)
            X_target_w_t_left_node=X_target_w_t_subtree[X_target_w_t_subtree[:,feature]<split_value,:]
            X_target_w_t_right_node=X_target_w_t_subtree[X_target_w_t_subtree[:,feature]>=split_value,:]
            entropy_after=conditionalEntropy(X_target_w_t_left_node[:,-2:],X_target_w_t_right_node[:,-2:])
            if entropy_after<best_split_entropy_after:
                best_split_entropy_after=entropy_after
                best_split_value=split_value
    #print "for feature",feature, "best entropy_after",best_split_entropy_after
    #print "entropy before and after",entropy_before, best_split_entropy_after
    IG=entropy_before-best_split_entropy_after
    return IG,best_split_value

def entropy(target_w_t_node):
    w_t_neg=np.sum(target_w_t_node[target_w_t_node[:,0]==-1,1])
    w_t_pos=np.sum(target_w_t_node[target_w_t_node[:,0]==1,1])
    w_t_twoclasses=float(w_t_neg+w_t_pos)

    if w_t_twoclasses==0: return 0
    p_neg=w_t_neg/w_t_twoclasses
    p_pos=w_t_pos/w_t_twoclasses
    product_pos=0 if p_pos==0 else p_pos*np.log2(p_pos)
    product_neg=0 if p_neg==0 else p_neg*np.log2(p_neg)
    entropy=-(product_pos+product_neg)
    return entropy


def conditionalEntropy(target_w_t_left_node,target_w_t_right_node):
    entropy_left=entropy(target_w_t_left_node)
    entropy_right=entropy(target_w_t_right_node)
    #if entropy_left==0: print "entropy_left=0"
    #if entropy_right==0: print "entropy_right=0"
    w_t_left_node=target_w_t_left_node[:,-1]
    w_t_right_node=target_w_t_right_node[:,-1]
    w_t_left_sum=np.sum(w_t_left_node)
    w_t_right_sum=np.sum(w_t_right_node)
    w_t_twonodes=float(w_t_left_sum+w_t_right_sum)
    if w_t_twonodes==0: return 0
    p_left=w_t_left_sum/w_t_twonodes
    p_right=w_t_right_sum/w_t_twonodes
    entropy_after=p_left*entropy_left+p_right*entropy_right
    return entropy_after

def crossValidation(X_target,B,k):
    N=X_target.shape[0]
    folder_size=N/k
    size_of_B=len(B)
    e_t_test=np.zeros((k,size_of_B))
    e_t_train=np.zeros((k,size_of_B))

    for folder_index in range(k):
        #e_t_m=np.zeros((size_of_M,))
        for b_index,b in enumerate(B):
            folder_start=folder_size*folder_index
            folder_end=folder_size*folder_index+folder_size
            X_target_test=X_target[folder_start:folder_end,:]
            X_target_train=np.vstack((X_target[:folder_start,:],X_target[folder_end:,:]))
            boosted=boosting(X_target_train,b)
            e_t_test[folder_index,b_index]=classifyAndError(X_target_test,boosted)
            e_t_train[folder_index,b_index]=classifyAndError(X_target_train,boosted)
        #e_t[folder_index,]=np.mean(e_t_m)
    print "train set error rate matrix (size_of_folder*size_of_B):"
    print "row represents folder=",range(k), ", column represents base classifier=",B
    print e_t_train
    print "train set average error rate:"
    print np.mean(e_t_train,0)
    print "train set std:"
    print np.std(e_t_train,0)

    print "test set error rate matrix (size_of_folder*size_of_B):"
    print "row represents folder=",range(k), ", column represents base classifier=",B
    print e_t_test
    print "test set average error rate:"
    print np.mean(e_t_test,0)
    print "test set std:"
    print np.std(e_t_test,0)

def myABoost(filename,B, k):
    X, target50,target75=readCsvToBinary(filename)
    X_target50=np.hstack((X,target50[:,None]))
    print "boston50 crossValidation result:"
    crossValidation(X_target50,B,k)
    X_target75=np.hstack((X,target75[:,None]))
    print "boston75 crossValidation result:"
    crossValidation(X_target75,B,k)

B=range(1,11)
myABoost(filename,B,10)
