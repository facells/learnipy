# learnipy
making machine learning and deep learning accessible to everyone

written with ♥ by Fabio Celli, 

email: fabio.celli.phd@gmail.com

twitter: @facells

tested in Google colab

License: MIT (Commercial use,  Modification, Distribution, Private use are permitted, Liability is yours, No software warranty)



===USAGE=== 

 1. to train a model: load code and data (.csv or .zip) in colab and type

 %run learnipy.py 'options' traindata [testdata]

 

 2. to make predictions on new data: load model (.h5) and data (.csv or .zip) in colab and type

 %run learnipy.py '-d.pred' model.h5 [testdata]



===DATA FORMATTING===

 data.csv must be a , separated file, 

 the target column must be named 'class'

 the text column must be named 'text'

 

 data.zip must contain .png .jpg files.

 the files names must be , separated. example: imgID,class,.jpg



===OPTIONS===

 ===data management===

 -d.c=l      manually define type of target class. params: l=label, 0=number

 -d.gen=132  generate dataset, create gen.csv.params 1=num instances x1000, 3=num features x10, 2=num informative features x10

 -d.viz      visualize class distribution and pca-projected 2d data scatterplot

 -d.fdst     print info on feature distribution

 -d.data     show preview of processed data

 -d.cnorm    normalize numeric class

 -d.save     save model as .h4 (machine learning) or .h5 (deep learning) file

 -d.pred     use model to make predictions on new data

 ===preprocessing===

 -p.rand     randomize instances in the training set

 -p.norm     feature normalization, range 0,1 (applied by default with sgd and nb)

 -p.tl       text to lowercase

 -p.tc       clean text from non alphanum char and multiple spaces

 ===feature reduction===

 -r.svd=5    singular value decomposition. turn sparse label binarized matrix to dense and sync. param: 5=number of features

 -r.lsa=5    latent semantic analysis. turn sparse word/char matrix to dense and sync. param: 5=number of features

 ===text feature extraction===

 -x.ng=23cf  ngrams. turn text to word|char ngrams freq|tfidf matrix and apply lda. params: 2=min, 3=max, c=chars|w=words, f=freq|t=tfidf

 -x.tm=5     token matrix. turn text into word frequency matrix. 5=integer, number of features

 -x.ts=5     token sequences. turn text into sequence word index feature matrix. columns are padded sequences of words. 5=integer, number of features 

 -x.w2v      extract word2vec dictionary and save it. also visualize a 2d word2vec space

 -x.d2v=5    turn text into doc2vec dense feature matrix. 5=integer, number of features

 -x.bert     turn text into multi-language bert 768-length vectors

 -x.d=d.lex  turn text into vectors from custom dictionary dimensions. d.lex is an external lexical resource, comma separated

 ===unsupervised learning===

 -u.km=2     processed feature analysis with kmeans centroid clustering. add a new colum to dataset. results in analysis.txt. params: 2=num clusters

 -u.optics   processed feature analysis with optics density clustering. add a new colum to dataset. results in analysis.txt. 

 -u.msh      processed feature analysis with mshift density clustering. add a new colum to dataset. results in analysis.txt.

 -u.arm      association rule mining. default algorithm: apriori. prints results in analysis.txt

 -u.corr     processed feature analysis with pearson correlations. prints results in analysis.txt

 ===supervised learning===

 -s.base     majority baseline for classification and regression

 -s.nb       probabilistic models. complement naive bayes for classification, bayes ridge for regression

 -s.lr       linear regression and logistic regression

 -s.lcm      linear combination models, linear discriminant classifiction and partial least squares regression

 -s.sgd      linear modeling with stochastic gradient descent

 -s.knn      k nearest neighbors classification and regression

 -s.dt       decision trees and regression trees

 -s.mlp      multi layer perceptron

 -s.svmp=3   svm with poly kernel. param p=polynomial kernel. 3=dimensions of polynomial kernel, 

 -s.svm      svm with radial basis function.

 -s.rf       ensemble learning, random forest

 -s.ada      ensemble learning, adaboost based on samme.r algorithm

 -s.xgb      ensemble learning, xgboost

 -s.nn=f     deep learning. params: f=feedfwd|i=imbalance|r=rnn|l=lstm|b=bilstm|g=gru|c=cnn. 

 ===evaluation===

 -e.tts=0.2  train-test split. params 0.2=20% test split. param not valid if test set is provided.
