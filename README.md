'''
# LEARNIPY
* version 0.9
* making machine learning easy for everyone
* written with ♥ by Fabio Celli, 
* email: fabio.celli.phd@gmail.com
* twitter: @facells
* tested in Google colab
* license: MIT (Commercial use,  Modification, Distribution permitted)
* conditions: Report the following license and copyright notice with code.
* warranty: Liability is yours, No software warranty


"Copyright (c) 2021 Fabio Celli.
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."

### 1) AIM 
* we want to make machine learning accesible and easy to use for learners. 
* design a system that is self-contained (one file), portable, 100% written in Python.

### 2) DATA FORMATTING
* data.csv must be a comma-separated file (,)
* the target column can be named 'class' in the .csv file or defined with -d.c= option
* the text column can be named 'text' in the .csv file or defined with -d.s= option
* data.zip must contain .png or .jpg files. named like the example: imgID,class,.jpg

### 4) USAGE 
* to train a model: %run learnipy.py 'options' traindata [testdata], for example

 >%run learnipy.py '-d.t=c -x.tm=700 -d.viz -s.nn=f' traindata.csv

* 'options' is a string containing the operations, defined at paragraph 4.
* yourdata.csv can be a .csv for tabular and text data or .zip for pictures.
* [testdata] is optional, if given is used as a test set, if not the training set is split
* to make predictions on new data: %run learnipy.py '-d.pred' model testdata, for example

 >%run learnipy.py '-d.pred' model.h5 testdata.csv

* models can have .h5 (deep learning) or .h4 (machine learning) extension
* try it on https://colab.research.google.com/drive/1DfDp2VFaTTMz_B6uLrOdWKQkrer32S9M?usp=sharing

### 4) DOCUMENTATION
#### data management
* -d.t=c|r    *define type of task. c=classification, r=regression*
* -d.x=n,m,o  *define the columns to exclude. n,m,o=names of columns to exclude*
* -d.k=n,m,o  *define the columns to keep. n,m,o=names of columns to keep*
* -d.s=n      *define the string column treated as text. n=name of text column
* -d.c=n      *define the column of the target class. n=name (for .csv) or index (for .zip)* 
* -d.r=0      *do not use feature reduction (not applicable with -d.save)*
* -d.f=c_v    *filter. keep only rows of column c with value v*
* -d.b=0.5    *resample rows. if <1 subsample. if >1 bootstrapping with duplication *
* -d.m=1      *fill class missing values with mean/mode (otherwise are deleted by default)*
* -d.g=c_a|s  *group rows by column c as a=average or s=sum*
* -d.viz      *print pca-projected 2d data scatterplot and other visualizations*
* -d.md       *model details. prints info on algorithm parameters and data modeling*
* -d.fdst     *print info on feature distribution*
* -d.data     *show preview of processed data*
* -d.save     *save model as .h4 (machine learning) or .h5 (deep learning) file*
* -d.pred     *use model to make predictions on new data*
* -d.export=f *export processed data in csv. f=filename.csv*
#### process mining
* -m.pnam[=k] *petri net from alpha miner algorithm. k=20 filter top 20 variants*
* -m.hnas[=k] *heuristic net from A-star pathfinding algorithm. k=20 filter top 20 variants*
* -m.bpmn[=k] *bpmn from inductive miner algorithm. k=20 filter top 20 variants*
#### preprocessing
* -p.ir       *instance position randomization*
* -p.cn       *class normalize. turn numeric class to range 0-1*
* -p.fn       *feature normalize range 0-1 (applied by default with some nn, sgd and nb)*
* -p.tl       *text to lowercase*
* -p.tc       *text cleaning. removes non alphanum char and multiple spaces*
* -p.trs      *text regex stopwords. removes words from length 1 to length 3*
* -p.tsw=a,b  *text stopwords. removes stopwords, a,b=stopwords list, no spaces allowed.*
#### feature reduction
* -r.svd=5    *turn sparse label matrix to dense and sync. 5=number of features*
* -r.lsa=5    *turn sparse word/char matrix to dense and sync. 5=number of features*
#### feature extraction
* -x.ng=23cf4 *text ngrams. 2=min, 3=max, c=chars|w=words, f=freq|t=tfidf, 4=num x 100*
* -x.tm=5     *text token matrix. 5=number of features*
* -x.ts=5     *text token sequences. 5=number of features* 
* -x.cm=5     *text char matrix. 5=number of features*
* -x.bert     *text multilang BERT. 768 features*
* -x.zsl=l,l  *text zero shot LLM label prediction. l,l=labels comma separated*
* -x.d=e      *text featurs from custom dictionary.check learnipy/resources*
* -x.rsz[=32] *image resize. 32=size 32x32, default 16x16 (768 features)*
* -x.resnet   *image resnet model. 2048 features from pre-trained model*
* -x.vgg      *image vgg model. 512 sparse features from pre-trained model*
* -x.effnet   *image effnet model. 1408 dense features from pre-trained model*
#### unsupervised learning
* -u.km=2     *kmeans centroid clustering. add 1 column. 2=num clusters*
* -u.kmpp=2   *kmeans++ centroid clustering. add 1 column. 2=num clusters*
* -u.sc=2     *spectral clustering. add 1 column. 2=num clusters*
* -u.optics   *optics density clustering. one cluster for noise add 1 column*
* -u.msh      *mshift, density clustering. add 1 column*
* -u.ap       *affinity propagation exemplar clustering.add 1 column.*
* -u.som      *self organising map, neural network clustering. add 1 column.*
* -u.arl      *association rule learning with apriori.*
* -u.corr=s|p *correlation rankings. s=spearman (monotone+linear), p=pearson (linear)*
* -u.corm=s|p *correlation matrix. s=spearman (monotone+linear), p=pearson (linear)*
#### outlier detection
* -o.if       *isolation forest. remove outlier rows random forest regions*
* -o.mcd      *minimum covariance determinant. remove outlier rows with gaussian distr*
* -o.lof      *local outlier factor. remove outlier rows with optics less dense regions*
#### supervised learning
* -s.base     *majority baseline for classification and regression*
* -s.nb       *probability models. naive bayes, bayesian ridge*
* -s.lr       *linear regression and logistic regression*
* -s.lcm      *linear combination models. linear discriminant classif, partial least squares*
* -s.sgd      *linear modeling with stochastic gradient descent*
* -s.knn      *k nearest neighbors classification and regression*
* -s.dt[=1]   *decision/regression trees. 1=pruning level*
* -s.mlp      *multi layer perceptron*
* -s.svm[=p3] *svm. p=polynomial kernel|r=rbf kernel (default), 3=kernel degrees*
* -s.rf       *ensemble learning, random forest*
* -s.ada      *ensemble learning, adaboost based on samme.r algorithm*
* -s.xgb      *ensemble learning, xgboost*
* -s.nn=f     *neural nets. f=feedfwd|i=imbalance|r=rnn|l=lstm|b=bilstm|g=gru|c=cnn*
#### time series forecasting
* -t.arma     *auto regression moving average*
* -t.arima    *auto regression integrated moving average*
* -t.sarima   *seasonal auto regression integrated moving average*
* -t.hwes     *Holt-Winters exponential smoothing*
#### evaluation
* -e.tts=0.2  *train-test split. 0.2=20% test split. ignored if test set is provided*

### 5) CHANGELOG
* v0.0: developed the main features
* v0.1: added -u.corr, -u.arl, -x.w2v, -x.d2v, -s.sgd, -s.xgb, .zip input, -s.nn=c
* v0.2: added -x.bert, -x.tm, -x.ts, improved -s.nn, removed -e.cv (cross validation)
* v0.3: improved -x.bert, -x.d and -d.viz, added -d.c, -d.s, -d.m, changed -d.gen to -g.d
* v0.4: added -d.export -g.mct, -u.som, -d.md, included -s.psvm in -s.svm, moved -u.w2v
* v0.5: added -p.trs, -p.tsw, -o.if, -o.mcd, -o.lof, -u.ap, fixed bug on .zip reading
* v0.6: improved anomaly detection evaluation, added -t., -x.mobert
* v0.7: added -x.effnet, -x.resnet, -x.vgg, -x.rsz, improved -u.corr, -x.ng
* v0.8: added -u.corr and -u.corm, , -d.f, -d.g, -d.k, -d.b, removed w2v and d2v
* v0.9: added -x.zsl, -u.kmpp, -u.sc, improved -d.viz, removed -x.mobert
* v0.10: fixed -s.dt, added process mining, dropped generate data

### 6) TO DO LIST
* add agent based models
* explainable AI
* add network analysis
* add forecasting with sktime
* improve test set input

'''


'''
