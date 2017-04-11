import numpy as np

def readCsvToBinary(filename):
    file = np.genfromtxt(filename, delimiter=',')
    X=file[:,:-1]
    target=file[:,-1]
    tao50= np.median(target)
    target50=np.copy(target)
    tao75=np.percentile(target,75)
    target75=np.copy(target)
    target50[target50>=tao50]=1
    target50[target50!=1]=0
    target75[target75>=tao75]=1
    target75[target75!=1]=0

    return X, target50,target75

def bootstrap(X_target):
    N=X_target.shape[0]
    index_k=np.random.randint(N, size=N)
    X_target_k= X_target[index_k,:]
    return X_target_k

def generateForest(X_target,m):
    N=X_target.shape[0]
    forest=[]
    for k in range(100):
        X_target_k=bootstrap(X_target)
        feature_split_tree_k,leaf_class_k=generateTree(X_target_k,m)
        forest.append((feature_split_tree_k,leaf_class_k))
    return forest

def classifyAndError(X_target,forest):
    error_t=0.
    N=X_target.shape[0]
    class_t=np.zeros((N,))
    for i in range(N):
        class_t_vote=np.zeros((100,))
        for tree_index,feature_split_leaf_class in enumerate(forest):
            feature_split_tree=feature_split_leaf_class[0]
            leaf_class=feature_split_leaf_class[1]
            if X_target[i,feature_split_tree[0][0]]<feature_split_tree[0][1]:
                if X_target[i,feature_split_tree[1][0]]<feature_split_tree[1][1]:
                    class_t_vote[tree_index,]=leaf_class[0]
                else:
                    class_t_vote[tree_index,]=leaf_class[1]
            else:
                if X_target[i,feature_split_tree[2][0]]<feature_split_tree[2][1]:
                    class_t_vote[tree_index,]=leaf_class[2]
                else:
                    class_t_vote[tree_index,]=leaf_class[3]
        #print class_t_vote[class_t_vote==0,].shape[0],class_t_vote[class_t_vote==1,].shape[0]
        if class_t_vote[class_t_vote==0,].shape[0]<=class_t_vote[class_t_vote==1,].shape[0]:
            class_t[i,]=1
        else:
            class_t[i,]=0
        #print class_t[i,], X_target[i,-1]
        if class_t[i,]!=X_target[i,-1]:
            error_t+=1.
    #print "N", N
    e_t=error_t/N
    return e_t

def generateTree(X_target,m):
    feature_root,split_value_root=generateLayer(X_target,m)
    X_target_left_node=X_target[X_target[:,feature_root]<split_value_root,:]
    X_target_right_node=X_target[X_target[:,feature_root]>=split_value_root,:]

    feature_left,split_value_left= generateLayer(X_target_left_node,m)
    target_left_left=X_target_left_node[X_target_left_node[:,feature_left]<split_value_left,-1]
    target_left_right=X_target_left_node[X_target_left_node[:,feature_left]>=split_value_left,-1]
    left_left_class=decideLeaf(target_left_left)
    left_right_class=decideLeaf(target_left_right)

    feature_right,split_value_right= generateLayer(X_target_right_node,m)
    target_right_left=X_target_right_node[X_target_right_node[:,feature_right]<split_value_right,-1]
    target_right_right=X_target_right_node[X_target_right_node[:,feature_right]>=split_value_right,-1]
    right_left_class=decideLeaf(target_right_left)
    right_right_class=decideLeaf(target_right_right)

    feature_split_tree=[(feature_root,split_value_root),(feature_left,split_value_left),(feature_root,split_value_right)]
    leaf_class=[left_left_class,left_right_class,right_left_class,right_right_class]
    return feature_split_tree,leaf_class

def decideLeaf(target_leaf):
    count_pos_leaf=target_leaf[target_leaf==1].shape[0]
    count_neg_leaf=target_leaf[target_leaf==0].shape[0]
    if count_neg_leaf<=count_pos_leaf:
        return 1
    else:
        return 0

def generateLayer(X_target_subtree,m):
    max_IG_feature=-np.inf
    feature_subset=np.random.randint(13, size=m)
    for feature in feature_subset:
#for earch feature, each find the split with max informaiton gain within feature
#second, find the max information gain across all features
        IG_feature,feature_split_value=informationGain(X_target_subtree,feature)
        if IG_feature>max_IG_feature:
            best_feature=feature
            best_feature_split_value=feature_split_value
    return best_feature,best_feature_split_value

def entropy(target_node):
    count_pos=target_node[target_node==1,].shape[0]
    count_neg=target_node[target_node==0,].shape[0]
    #print "count_pos_neg",count_pos,count_neg
    count_twoclasses= count_neg + count_pos
    if count_twoclasses==0: return 0
    p_pos=float(count_pos)/count_twoclasses
    p_neg=float(count_neg)/count_twoclasses
    #print "p",p_pos,p_neg
    product_pos=0 if p_pos==0 else p_pos*np.log2(p_pos)
    product_neg=0 if p_neg==0 else p_neg*np.log2(p_neg)
    entropy=-(product_pos+product_neg)
    return entropy

def conditionalEntropy(target_left_node,target_right_node):
    entropy_left=entropy(target_left_node)
    entropy_right=entropy(target_right_node)
    #print "left_entropy",entropy_left,"right_entropy",entropy_right
    count_left=target_left_node.shape[0]
    count_right=target_right_node.shape[0]
    count_twonodes=count_left+count_right
    #print count_left,count_right
    if count_twonodes==0: return 0
    p_left=float(count_left)/count_twonodes
    p_right=float(count_right)/count_twonodes
    entropy_after=p_left*entropy_left+p_right*entropy_right
    return entropy_after

def informationGain(X_target_subtree,feature):
    entropy_before=entropy(X_target_subtree[:,-1])
    if feature==3:
        X_target_left_node=X_target_subtree[X_target_subtree[:,feature]==0]
        X_target_right_node=X_target_subtree[X_target_subtree[:,feature]==1]
        best_split_value=1
        best_split_entropy_after=conditionalEntropy(X_target_left_node[:,-1],X_target_right_node[:,-1])
    else:
        best_split_entropy_after=np.inf
        for split_quantile in range(10,100,10):
            split_value=np.percentile(X_target_subtree[:,feature],split_quantile)
            #print "feature",feature,"quantile",split_quantile,"split",split_value
            X_target_left_node=X_target_subtree[X_target_subtree[:,feature]<split_value,:]
            X_target_right_node=X_target_subtree[X_target_subtree[:,feature]>=split_value,:]
            #print "min",np.min(X_target_subtree[:,feature])
            #print "subtree",X_target_subtree.shape[0],"leftnode",X_target_left_node[:,-1].shape,"rightnode",X_target_right_node.shape
            entropy_after=conditionalEntropy(X_target_left_node[:,-1],X_target_right_node[:,-1])
            #print entropy_after, best_split_entropy_after
            if entropy_after<=best_split_entropy_after:
                best_split_entropy_after=entropy_after
                best_split_value=split_value
                #print "best_split_value=",best_split_value
    IG=entropy_before-best_split_entropy_after
    #print best_split_value
    return IG, best_split_value

def myRForest(filename,M,k):
    X, target50,target75=readCsvToBinary(filename)
    X_target50=np.hstack((X,target50[:,None]))
    print "boston50 crossValidation result:"
    crossValidation(X_target50,M,k)
    #print "boston50 error rate "
    X_target75=np.hstack((X,target75[:,None]))
    print "boston75 crossValidation result:"
    crossValidation(X_target75,M,k)

def crossValidation(X_target,M,k):
    N=X_target.shape[0]
    folder_size=N/k
    size_of_M=len(M)
    e_t_test=np.zeros((k,size_of_M))
    e_t_train=np.zeros((k,size_of_M))

    for folder_index in range(k):
        #e_t_m=np.zeros((size_of_M,))
        for m_index,m in enumerate(M):
            folder_start=folder_size*folder_index
            folder_end=folder_size*folder_index+folder_size
            X_target_test=X_target[folder_start:folder_end,:]
            X_target_train=np.vstack((X_target[:folder_start,:],X_target[folder_end:,:]))
            forest=generateForest(X_target_train,m)
            e_t_test[folder_index,m_index]=classifyAndError(X_target_test,forest)
            e_t_train[folder_index,m_index]=classifyAndError(X_target_train,forest)
        #e_t[folder_index,]=np.mean(e_t_m)

    print "train set error rate matrix (size_of_folder*size_of_M):"
    print "row represents folder=",range(k), ", column represents M=",M
    print e_t_train
    print "train set average error rate:"
    print np.mean(e_t_train,0)
    print "train set std:"
    print np.std(e_t_train,0)    #return X_target_train,X_target_test

    print "test set error rate matrix (size_of_folder*size_of_M):"
    print "row represents folder=",range(k), ", column represents M=",M
    print e_t_test
    print "test set average error rate:"
    print np.mean(e_t_test,0)
    print "test set std:"
    print np.std(e_t_test,0)


M=range(1,14)
myRForest(filename,M, 10)
