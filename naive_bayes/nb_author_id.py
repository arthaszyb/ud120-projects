#!/usr/bin/python
#coding:utf-8
""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

def classify(x,y,z,w):
    '''朴素贝叶斯算法，输入训练样本，训练标签，测试样本，测试标签（用于测试准确度），输出预测结果和预测准确度'''
    clf=GaussianNB()
    _time0=time()
    clf.fit(x,y)
    print "fit used %ss" % round(time() - _time0,3)
    _time1=time()
    pre_res=clf.predict(z)
    print "predict used %ss" % round(time() - _time1,3)
    pre_score=clf.score(z,pre_res)
    return pre_res,pre_score

if __name__ == '__main__':
    pre_res, pre_score = classify(features_train, labels_train,features_test,labels_test)
    print "the predict result is %s" % pre_res
    print "predict score is %f" % pre_score
#########################################################


