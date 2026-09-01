documentation='''
# LEARNIPY
* version 0.13
* making data science easier
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
* -d.f=c_v    *filter. keep only rows of column c with value v*
* -d.b=0.5    *resample rows. if <1 subsample. if >1 bootstrapping with duplication *
* -d.m=1      *fill class missing values with mean/mode (otherwise are deleted by default)*
* -d.g=c_a|s  *group rows by column c as a=average or s=sum*
* -d.viz      *print pca-projected 2d data scatterplot and other visualizations*
* -d.r        *reverse lines in data*
* -d.md       *model details. prints info on algorithm parameters and data modeling*
* -d.fdst     *print info on feature distribution*
* -d.data     *show preview of processed data*
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
* -p.trs      *text regex strip. removes words from length 1 to length 3*
* -p.tsw=a,b  *text stop words. removes stopwords, a,b=stopwords list, no spaces allowed.*
* -p.lda=5    *keeps only topical words using latent dirichlet allocation. 5=words per topic*
#### feature reduction
* -r.svd=5    *turn sparse label matrix to dense. 5=number of features*
* -r.lsa=5    *turn sparse word/char matrix to dense. 5=number of features*
#### feature extraction
* -x.ng=23cf4 *text ngrams. 2=min, 3=max, c=chars|w=words, f=freq|t=tfidf, 4=num x 100*
* -x.tm=5     *text token matrix. 5=number of features*
* -x.ts=5     *text token sequences. 5=number of features* 
* -x.cm=5     *text char matrix. 5=number of features*
* -x.trans=b  *text transformers. b=BERT|bl=BERTlarge|bix=BERTitalian|r=RoBERTa. 768 features*
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
* -u.dcs      *document cosine similarity anlysis with word embeddings*
* -u.irr      *inter-rater reliability scores*
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
* -s.nn=f     *neural nets. f=feedfwd|i=imbalance|o=output|r=rnn|l=lstm|b=bilstm|g=gru|c=cnn*
#### time series forecasting
* -t.arima    *auto regression integrated moving average*
* -t.sarima   *seasonal auto regression integrated moving average*
* -t.hwes     *Holt-Winters exponential smoothing*
#### agent based modeling
* -a.ss=i,f   *shelling segregation simulation. i=grid (int), f=probability (float)*
* -a.wd=i     *boltzman wealth distribution simulaiton. i=population (int)*
* -a.s=i,i    *sugarscape life simulation. i=grid (int), i=agents (int)*
* -a.sir=i,p  *SIR infection spread simulation. i=population (int), p=probability of infection (float)*
* -a.mf=i,p   *Multipath forecasting simulation. i=population (int), p=social mobility (float)*
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
* v0.10: fixed -s.dt, added process mining, transformers. removed generate data, shuffle on -e.tts
* v0.11: fixed -p.trs, added -p.lda, -u.dcs, -u.irr
* v0.12: added feature importance in -s algorithms, removed -d.pred, -d.r=0, reduction by default
* v0.13: added agent based models, added -d.r

### 6) TO DO LIST
* permutation feature importance
* genetic algorithms
* add network analysis
* add forecasting with sktime
* improve test set input


'''

print("\n\n");
print('---START PROCESS---');
print('import and install libraries')

import os;
#os.system('pip install -U tensorflow==2.8.0')
#os.system('pip install --upgrade tensorflow-hub')

import warnings; warnings.filterwarnings('ignore'); 
import datetime as DT;
import pandas as PD;
import numpy as NP;
import tensorflow as TF;
import zipfile as ZF;
import sklearn as SK; from sklearn import *; 
from skimage.io import imread; from skimage.transform import resize;
import matplotlib.pyplot as MP; MP.rcParams["figure.figsize"]=(5,4);
#import gensim as W2V; from gensim.models.doc2vec import Doc2Vec, TaggedDocument;
import nltk; from nltk.tokenize import word_tokenize; nltk.download('punkt', quiet=True);
import sys; 
import re;
import math;
import joblib;
import shutil;
import tensorflow_hub as TH;
from tqdm import tqdm;
from io import StringIO;
import urllib.request; 
import statsmodels.api as SM
from scipy import stats as ST



NP.random.seed(1); TF.random.set_seed(2);
TF.compat.v1.logging.set_verbosity(TF.compat.v1.logging.ERROR);
MP.clf();


timestamp=DT.datetime.now(); print(f"time:{timestamp}");

#---system option reading
o=sys.argv[1]+' ';

if '-h' in o: 
 print(documentation); print('---END PROCESS---'); sys.exit();

#import image pretrained models
if '-x.resnet' in o:
 print('import resnet to extract 2048 features from images');
 from tensorflow.keras.applications.resnet50 import ResNet50
 from tensorflow.keras.preprocessing import image
 from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
 from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
 from tensorflow.keras.models import Model
 imgmodel = ResNet50(weights='imagenet')
 base_model = ResNet50(weights='imagenet', include_top=False)
 # add a global spatial average pooling layer
 x = base_model.output
 head = GlobalAveragePooling2D()(x)
 imgmodel = Model(inputs=base_model.input, outputs=head)   

if '-x.vgg' in o:
 print('import vgg to extract 512 features from images');
 from tensorflow.keras.applications.vgg16 import VGG16
 from tensorflow.keras.preprocessing import image
 from tensorflow.keras.applications.vgg16 import preprocess_input
 from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
 from tensorflow.keras.models import Model
 imgmodel = VGG16(weights='imagenet')
 base_model = VGG16(weights='imagenet', include_top=False)
 # add a global spatial average pooling layer
 x = base_model.output
 head = GlobalAveragePooling2D()(x)
 imgmodel = Model(inputs=base_model.input, outputs=head)  


if '-x.effnet' in o:
 print('import efficientnet to extract 1408 features from images');
 from tensorflow.keras.applications import EfficientNetB2
 from tensorflow.keras.preprocessing import image
 from tensorflow.keras.applications.efficientnet import preprocess_input
 from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
 from tensorflow.keras.models import Model
 imgmodel = EfficientNetB2(weights='imagenet')
 base_model = EfficientNetB2(weights='imagenet', include_top=False)
 # add a global spatial average pooling layer
 x = base_model.output
 head = GlobalAveragePooling2D()(x)
 imgmodel = Model(inputs=base_model.input, outputs=head)  




#---files reading
try:
 f=sys.argv[2]; #take dataset or model
except:
 print('no file loaded. generating gen.csv'); #inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();
 if not '-g.d=' in o:
  o=o+' -g.d=123'; 
#---generate data
 if '-g.d=' in o: #generate data with a normal distribution filled with random noise
  r_=re.findall(r'-g.d=(.+?) ',o); 
  ns=int(r_[0][0]); nf=int(r_[0][1]); ni=int(r_[0][2]); 
  ns=ns*1000; nf=nf*10; ni=ni*10; nr=nf-ni; f='gen.csv';
  print(f'generating dataset with {ns} samples and {nf} features, {ni} informative');
  x_, y_ = SK.datasets.make_classification(n_samples=ns, n_features=nf); 
  x_=PD.DataFrame(x_); y_=PD.DataFrame(y_, columns=['class']); 
  g_=PD.concat([x_, y_], axis=1); g_.to_csv('gen.csv', sep=',', encoding='utf-8'); 
  
try:
 f2=sys.argv[3]; #take testset or newdataset
except:
 print('using training set');




#if '-g.mct' in o: #markov chains to generate text with next word probability 
# print('generating sentence from training data:');
# os.system('pip install markovify'); import markovify;
# ft=open(f, encoding='utf8').read();
# #text_model = markovify.NewlineText(ft, state_size = 2); 
# #for poetry (d.headline_text= pandas serie of newline text)
# text_model = markovify.Text(ft, state_size=3); 
# print(text_model.make_sentence()); inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();
# #print(text_model.make_short_sentence(280));


datatype='';


#---initialize default settings
if 'f2' in locals() and '.csv' in f2: #import csv test set
 datatype='csv'; 

if 'f2' in locals() and '.zip' in f2:
 datatype='zip';

if '-x.rsz' in o:#extract features from image with resize
 r_=re.findall(r'-x.rsz=(.+?) ',o); size=int(r_[0]); imgfeats=(size*size)*3; 
 print(f"using image size {size}x{size}, extract {imgfeats} features");  
else:#otherwise apply default size 28x28
 size=16; imgfeats=(size*size)*3;  

if '-d.ts=' in o:#get name of timestamp column
 r_=re.findall(r'-d.ts=(.+?) ',o); tscol=(r_[0]); 
 print(f"using '{tscol}' as timestamp column");
else:#otherwise apply default name
 tscol='date'; print('search "date" as timestamp column'); 

if '-d.s=' in o:#get name of string column
 r_=re.findall(r'-d.s=(.+?) ',o); txtcol=(r_[0]); 
 print(f"using '{txtcol}' as string column");
else:#otherwise apply default name
 txtcol='text'; print('search "text" as string column'); 

if '-d.c=' in o:#get name of class column
 if re.search(r'-d.c=[a-zA-Z]', o):
  r_=re.findall(r'-d.c=(.+?) ',o); tgtcol=(r_[0]); 
  print(f"'{tgtcol}' is the target class column"); 
 if re.search(r'-d.c=[0-9]', o):
  r_=re.findall(r'-d.c=(.+?) ',o); tgtcol=int(r_[0]); 
  print(f"'{tgtcol}' is the target class index"); 
else:#otherwise apply default name
 if '.csv' in f or datatype=='csv':
  tgtcol='class'; print('search "class" as target class'); 
 if '.zip' in f or datatype=='zip':
  tgtcol=1; print('extract index 1 from comma separated file name as target class');

if '-d.x=' in o:#get name of column to drop
 r_=re.findall(r'-d.x=(.+?) ',o); drop=r_[0].split(','); 

if '-d.k=' in o:#get name of column to keep
 r_=re.findall(r'-d.k=(.+?) ',o); keep=r_[0].split(','); 

if '-d.t=c' in o: #auto define classification or regression
 target='c'; print('task: classification')
if '-d.t=r' in o:
 target='r'; print('task: regression');



#---reading test set
if 'f2' in locals(): #import csv test set
 if '.csv' in f2:
  testname=f2.replace('.csv', ''); datatype='csv'; 
  x2_=PD.read_csv(f2, sep=',', encoding='utf8'); 
  testinst=len(x2_.index); print('using test set or new set');
 if '.zip' in f2:
  testname=f2.replace('.zip', ''); datatype='zip';
  zip=ZF.ZipFile(f2, 'r'); i_=zip.namelist(); print('using test set or new set');
  if ',' in i_[0]: #extract supervised data from files in zip
   x2_=[]; y2_=[]; names_=[];
   for i in i_:
    l_=i.split(','); 
    if l_[1]!='':
     label=l_[1]; 
    else: label=0;
    d=zip.open(i); #print(d);
    if '.jpg' in i or '.png' in i:
     d_=imread(d); names_.append(d.name); 
     d_=resize(d_, (size,size,3),anti_aliasing=True); dshape=(size,size); 
     d_=d_.flatten(); 
     x2_.append(d_); y2_.append(label); #read, resize and flatten images
 #ADD other file formats extraction
   x2_=NP.array(x2_).astype('float')/255.0; y2_=NP.array(y2_).astype('int');
   x2_=PD.DataFrame(x2_); y2_=PD.Series(y2_); testinst=len(x2_.index); 
   print(testinst);



#---import saved models
if '.h4' in f and '-d.pred' in o: #import machine learning saved model
 loadmodel = joblib.load(f);
 o=f.replace('-',' -'); x_=x2_; print(f"apply {o} to {f2}"); 
 #use model filename as o and test set as the main dataset 
 #to go into the pipeline with the same settings as the model trained
 if 'tgtcol' in locals() and tgtcol in x_.columns:
  y_=x_[tgtcol]; x_=x_.drop(columns=[tgtcol]); task='s'; print('target found');
 if 'txtcol' in locals() and txtcol in x_.columns:
  t_=x_[txtcol]; x_=x_.drop(columns=[txtcol]); print('text found');
 if 'tscol' in locals() and tscol in x_.columns:
  d_=x_[tscol]; x_=x_.drop(columns=[tscol]); print('date found');

if '.h5' in f and '-d.pred' in o: 
 loadmodel = TF.keras.models.load_model(f);
 o=f.replace('-',' -'); x_=x2_; print(f"apply {o} to {f2}"); 
 #use model filename as o and test set as the main dataset 
 #to go into the pipeline with the same settings as the model trained
 if datatype=='zip':
  x2_=PD.DataFrame(names_);
 if 'tgtcol' in locals() and tgtcol in x_.columns:
  y_=x_[tgtcol]; task='s'; print('target found');
  if datatype=='csv':
   x_=x_.drop(columns=[tgtcol]);
 if 'txtcol' in locals() and txtcol in x_.columns:
  t_=x_[txtcol]; x_=x_.drop(columns=[txtcol]); print('text found');
 if 'tscol' in locals() and tscol in x_.columns:
  d_=x_[tscol]; x_=x_.drop(columns=[tscol]); print('date found');



#---read data (if there is a test set create training+test set)
if '.csv' in f: 
 o=o.replace('-',' -'); print(f"processing {f} with {o}"); 
 filename=f.replace('.csv', ''); datatype='csv';
 x_=PD.read_csv(f, sep=',', encoding='utf8'); traininst=len(x_.index);
 dshape=[1];
 x_=x_.reset_index(drop=True); #start row index from 0
 if '-p.ri' in o:
  x_ = x_.sample(frac=1).reset_index(drop=True); 
  print('apply instance randomization in the training set'); #shuffle instances

#---concat train and test if separated
 if 'x2_' in locals():
  tts=((100/(traininst+testinst))*testinst)/100; 
  print(f'test set percentage={tts}');  
  if not x2_.empty:
   x_=PD.concat([x_, x2_], axis=0, ignore_index=True); 

 #---filter rows on one column value
 if '-d.f=' in o:
  r_=re.findall(r'-d.f=(.+?)_(.+?) ', o); #print(r_);
  fcol=r_[0][0]; fpat=r_[0][1];
  x_=x_.loc[x_[fcol] == fpat]; 
  print(f"dataset after filtering instances:\n\n {x_}\n");

#---subsample or bootstrap rows
 if '-d.b=' in o:
  r_=re.findall(r'-d.b=(.+?) ', o); #print(r_);
  nsamp=float(r_[0]); boot=False;
  if nsamp >1:
   boot=True
  x_=x_.sample(frac=nsamp, replace=boot, random_state=1); x_=x_.reset_index(); 
  x_=x_.drop(columns=['index']);
  print(f"dataset after resampling and randomize instances:\n\n {x_}\n");

#---group rows by one column
 if '-d.g=' in o:
  gbtype='c'; gbcol='class'
  r_=re.findall(r'-d.g=(.+?)_(.+?) ', o); #print(r_);
  gbcol=r_[0][0]; gbtype=r_[0][1];
  if gbtype=='s':
   x_=x_.groupby(gbcol).sum().reset_index(); gbout='sum'; 
  elif gbtype=='a':
   x_=x_.groupby(gbcol).mean().reset_index(); gbout='mean';
  elif gbtype=='j':
   x_=x_.groupby(gbcol)[txtcol].apply(lambda x_: ' '.join(x_)).reset_index(); gbout='join';
  elif gbtype=='c':
   x_=x_.groupby(gbcol).count().reset_index(); gbout='count';
  print(f"dataset after grouping instances with {gbout} function:\n\n {x_}\n");
  x_=x_.drop(gbcol, axis=1); 
  print(f"{gbcol} removed. \n\n ");

#---detect and separate different types of columns
 if 'tgtcol' in locals() and tgtcol in x_.columns: 
  #remove rows with missing values in class and extract target class dataframe
  if not '-d.m=1' in o:
   x_=x_.dropna(subset=[tgtcol]); 
   print(f"remove rows with missing values in {tgtcol}");
  y_=x_[tgtcol]; x_=x_.drop(columns=[tgtcol]); task='s'; print('target found');
 else:
  print('no class.\nfor supervised learning use -d.c=name or "class" column'); 
  task='u';

 if 'txtcol' in locals() and txtcol in x_.columns: 
  t_=x_[txtcol]; x_=x_.drop(columns=[txtcol]); 
  t_=t_.reset_index(drop=True); #start row index from 0
  print(f"taken {txtcol} as string column");

 if 'tscol' in locals() and tscol in x_.columns: 
  d_=x_[tscol]; x_=x_.drop(columns=[tscol]); 
  d_=d_.reset_index(drop=True); 
  d_=d_.rename({tscol:'date'}) #rename the time column
  print(f"taken {tscol} as date column");


if '.zip' in f: #extract data from .zip, loading in memory
 o=o.replace('-',' -'); print(f"processing {f} with {o}"); 
 filename=f.replace('.zip', ''); datatype='zip';
 zip=ZF.ZipFile(f, 'r'); i_=zip.namelist(); 
 if ',' in i_[0]: #extract supervised data from files in zip
  task='s'; print('target found, suppose supervised task');
  x_=[]; y_=[]; b_=[]; bl_=[]; print('reading data from .zip');
  if '-x.resnet' in o:
   print('apply resnet feature extraction');
  elif '-x.vgg' in o:
   print('apply vgg feature extraction');
  elif '-x.effnet' in o:
   print('apply efficientnet feature extraction');
  else:
   print(f'apply {size}x{size} img resize feature extraction');
  for num_i, i in enumerate(i_): #tqdm()
   l_=i.split(','); label=l_[tgtcol]; d=zip.open(i); #print(d);
   if '.jpg' in i or '.png' in i:
    if '-x.resnet' in o: #resnet img feature extraction
     d_=imread(d); d_=resize(d_, (224,224,3),anti_aliasing=True); 
     x = image.img_to_array(d_); b_.append(x); bl_.append(label);
     if len(b_) == 16 or num_i == len(i_)-1:
      x=NP.array(b_);
      x = preprocess_input(x)
      d_ = imgmodel.predict(x); dshape=(2048,); d_=d_.reshape((len(b_),2048));
      for id_ in range(len(b_)):
       x_.append(d_[id_]); y_.append(bl_[id_]);
      b_=[]; bl_=[];
    elif '-x.vgg' in o: #vgg img feature extraction
     d_=imread(d); d_=resize(d_, (224,224,3),anti_aliasing=True); 
     x = image.img_to_array(d_); b_.append(x); bl_.append(label);
     if len(b_) == 16 or num_i == len(i_)-1:
      x=NP.array(b_);
      x = preprocess_input(x)
      d_ = imgmodel.predict(x); dshape=(512,); d_=d_.reshape((len(b_),512));
      for id_ in range(len(b_)):
       x_.append(d_[id_]); y_.append(bl_[id_]);
      b_=[]; bl_=[];
    elif '-x.effnet' in o: #efficientnetB2 img feature extraction
     d_=imread(d); d_=resize(d_, (224,224,3),anti_aliasing=True); 
     x = image.img_to_array(d_); b_.append(x); bl_.append(label);
     if len(b_) == 16 or num_i == len(i_)-1:
      x=NP.array(b_);
      x = preprocess_input(x)
      d_ = imgmodel.predict(x); dshape=(1408,); d_=d_.reshape((len(b_),1408));
      for id_ in range(len(b_)):
       x_.append(d_[id_]); y_.append(bl_[id_]);
      b_=[]; bl_=[];

    else: #using resize img feature extraction (default)
     d_=imread(d); d_=resize(d_, (size,size,3),anti_aliasing=True); 
     dshape=(size,size); d_=d_.flatten();
     x_.append(d_); y_.append(label); 



#ADD other file formats extraction
  x_=NP.array(x_).astype('float')/255.0; x_=PD.DataFrame(x_); 
  traininst=len(x_.index);
  y_=PD.Series(y_);  #y_=y_.astype('float');

  if 'x2_' in locals() and not '-d.pred' in o:
   tts=((100/(traininst+testinst))*testinst)/100; 
   print(f'test set percentage={tts}');  #compute tts percentage  
   x_=PD.concat([x_, x2_], axis=0, ignore_index=True); 
   y_=PD.concat([y_, y2_], axis=0, ignore_index=True);

 else:  #extract unsupervised data from files in zip
  task='u'; 
  print('no class.\nfor supervised learning format your data as name,label,.jpg');
  x_=[];
  for i in i_:
   d=zip.open(i); #print(d);
   if '.jpg' in i or '.png' in i:
    d_=imread(d); dshape=d_.shape; 
    d_=d_.flatten(); x_.append(d_); #read and flatten images
#ADD other file formats extraction
  x_=NP.array(x_).astype('float')/255.0;
  x_=PD.DataFrame(x_);


if not '.zip' in f and not '.csv' in f and not '.zip' in f2 and not '.csv' in f2:
 print('please input a .csv or .zip dataset'); inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();


#---drop selected columns
if '-d.x=' in o:
 x_=x_.drop(columns=drop);

#---keep selected columns
if '-d.k=' in o:
 x_=x_[keep];

#---info on visualizations
if '-d.viz' in o:
 print('installing dim. reduction for visualization');
 os.system('pip install pacmap'); 
 import pacmap;
 pca=pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0); 
 #pca=SK.decomposition.PCA(2);
 print('all scatterplots are 2D PaCMAp-reduced spaces\ntheory: https://en.wikipedia.org/wiki/Dimensionality_reduction#PaCMAp');


if '-d.r' in o:
 print('reverse lines in data\n actual data:')
 x_ = x_[::-1] #reverse data
 print(x_)


#---unpreprocessed data ready

#---agent based models

if '-a.' in o:
 
 
 import os
 os.system('pip install -q agentpy')
 
 import agentpy as AP
 import networkx as NX
 import random
 import IPython
 from time import sleep
 from IPython.display import clear_output
 import matplotlib.pyplot as MP
 import seaborn as SB #sns
 import matplotlib as ML #mpl
 MP.rcParams['lines.linewidth'] = 2.0
 
 
 def three_frame(world, n_seq, seed=17):
     """Draw three timesteps.
 
     world: object with step, loop, and draw
     n_seq: 3-tuple, number of steps before each draw
     seed: random see for NumPy
     """
     NP.random.seed(seed)
     plt.figure(figsize=(10, 4))
 
     for i, n in enumerate(n_seq):
         plt.subplot(1, 3, i+1)
         world.loop(n)
         world.draw()
 
     plt.tight_layout()
 
 
 def savefig(filename, **options):
     """Save the current figure.
 
     Keyword arguments are passed along to plt.savefig
 
     https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
 
     filename: string
     """
     print("Saving figure to file", filename)
     plt.savefig(filename, **options)
 
 
 def underride(d, **options):
     """Add key-value pairs to d only if key is not in d.
 
     d: dictionary
     options: keyword args to add to d
     """
     for key, val in options.items():
         d.setdefault(key, val)
 
     return d
 
 
 def decorate(**options):
     """Decorate the current axes.
 
     Call decorate with keyword arguments like
 
     decorate(title='Title',
              xlabel='x',
              ylabel='y')
 
     The keyword arguments can be any of the axis properties
 
     https://matplotlib.org/api/axes_api.html
 
     In addition, you can use `legend=False` to suppress the legend.
 
     And you can use `loc` to indicate the location of the legend
     (the default value is 'best')
     """
     loc = options.pop("loc", "best")
     if options.pop("legend", True):
         legend(loc=loc)
 
     plt.gca().set(**options)
     plt.tight_layout()
 
 
 def legend(**options):
     """Draws a legend only if there is at least one labeled item.
 
     options are passed to plt.legend()
     https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
 
     """
     underride(options, loc="best", frameon=False)
 
     ax = plt.gca()
     handles, labels = ax.get_legend_handles_labels()
     if handles:
         ax.legend(handles, labels, **options)
 
 
 def set_palette(*args, **kwds):
     """Set the matplotlib color cycler.
 
     args, kwds: same as for sns.color_palette
 
     Also takes a boolean kwd, `reverse`, to indicate
     whether the order of the palette should be reversed.
 
     returns: list of colors
     """
     reverse = kwds.pop('reverse', False)
     palette = SB.color_palette(*args, **kwds)
 
     palette = list(palette)
     if reverse:
         palette.reverse()
 
     cycler = plt.cycler(color=palette)
     plt.gca().set_prop_cycle(cycler)
     return palette
 
 
 
 
 class Cell2D:
     """Parent class for 2-D cellular automata."""
 
     def __init__(self, n, m=None):
         """Initializes the attributes.
 
         n: number of rows
         m: number of columns
         """
         m = n if m is None else m
         self.array = NP.zeros((n, m), NP.uint8)
 
     def add_cells(self, row, col, *strings):
         """Adds cells at the given location.
 
         row: top row index
         col: left col index
         strings: list of strings of 0s and 1s
         """
         for i, s in enumerate(strings):
             self.array[row+i, col:col+len(s)] = NP.array([int(b) for b in s])
 
     def loop(self, iters=1):
         """Runs the given number of steps."""
         for i in range(iters):
             self.step()
 
     def draw(self, **options):
         """Draws the array.
         """
         draw_array(self.array, **options)
 
     def animate(self, frames, interval=None, step=None):
         """Animate the automaton.
 
         frames: number of frames to draw
         interval: time between frames in seconds
         iters: number of steps between frames
         """
         if step is None:
             step = self.step
 
         plt.figure()
         try:
             for i in range(frames-1):
                 self.draw()
                 plt.show()
                 if interval:
                     sleep(interval)
                 step()
                 clear_output(wait=True)
             self.draw()
             plt.show()
         except KeyboardInterrupt:
             pass
 
 
 def draw_array(array, **options):
     """Draws the cells."""
     n, m = array.shape
     options = underride(options,
                         cmap='Greens',
                         alpha=0.7,
                         vmin=0, vmax=1,
                         interpolation='none',
                         origin='upper',
                         extent=[0, m, 0, n])
 
     plt.axis([0, m, 0, n])
     plt.xticks([])
     plt.yticks([])
 
     return plt.imshow(array, **options)


 #SHELLING SEGREGATION (agents simulate residential patters, with a small preference to live near similar individuals)
 
 def locs_where(condition):
     """Find cells where a logical array is True.
 
     condition: logical array
 
     returns: list of location tuples
     """
     return list(zip(*NP.nonzero(condition)))
 
 from matplotlib.colors import LinearSegmentedColormap
 
 # make a custom color map
 palette = SB.color_palette('muted')
 colors = 'white', palette[1], palette[0]
 cmap = LinearSegmentedColormap.from_list('cmap', colors)
 
 from scipy.signal import correlate2d
 #from Cell2D import Cell2D, draw_array
 
 class Schelling(Cell2D):
     """Represents a grid of Schelling agents."""
 
     options = dict(mode='same', boundary='wrap')
 
     kernel = NP.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=NP.int8)
 
     def __init__(self, n, p):
         """Initializes the attributes.
 
         n: number of rows
         p: threshold on the fraction of similar neighbors
         """
         self.p = p
         # 0 is empty, 1 is red, 2 is blue
         choices = NP.array([0, 1, 2], dtype=NP.int8)
         probs = [0.1, 0.45, 0.45]
         self.array = NP.random.choice(choices, (n, n), p=probs)
 
     def count_neighbors(self):
         """Surveys neighboring cells.
 
         returns: tuple of
             empty: True where cells are empty
             frac_red: fraction of red neighbors around each cell
             frac_blue: fraction of blue neighbors around each cell
             frac_same: fraction of neighbors with the same color
         """
         a = self.array
 
         empty = a==0
         red = a==1
         blue = a==2
 
         # count red neighbors, blue neighbors, and total
         num_red = correlate2d(red, self.kernel, **self.options)
         num_blue = correlate2d(blue, self.kernel, **self.options)
         num_neighbors = num_red + num_blue
 
         # compute fraction of similar neighbors
         frac_red = num_red / num_neighbors
         frac_blue = num_blue / num_neighbors
 
         # no neighbors is considered the same as no similar neighbors
         # (this is an arbitrary choice for a rare event)
         frac_red[num_neighbors == 0] = 0
         frac_blue[num_neighbors == 0] = 0
 
         # for each cell, compute the fraction of neighbors with the same color
         frac_same = NP.where(red, frac_red, frac_blue)
 
         # for empty cells, frac_same is NaN
         frac_same[empty] = NP.nan
 
         return empty, frac_red, frac_blue, frac_same
 
     def segregation(self):
         """Computes the average fraction of similar neighbors.
 
         returns: fraction of similar neighbors, averaged over cells
         """
         _, _, _, frac_same = self.count_neighbors()
         return NP.nanmean(frac_same)
 
     def step(self):
         """Executes one time step.
 
         returns: fraction of similar neighbors, averaged over cells
         """
         a = self.array
         empty, _, _, frac_same = self.count_neighbors()
 
         # find the unhappy cells (ignore NaN in frac_same)
         with NP.errstate(invalid='ignore'):
             unhappy = frac_same < self.p
         unhappy_locs = locs_where(unhappy)
 
         # find the empty cells
         empty_locs = locs_where(empty)
 
         # shuffle the unhappy cells
         if len(unhappy_locs):
             NP.random.shuffle(unhappy_locs)
 
         # for each unhappy cell, choose a random destination
         num_empty = NP.sum(empty)
 
         for source in unhappy_locs:
             i = NP.random.randint(num_empty)
             dest = empty_locs[i]
 
             # move
             a[dest] = a[source]
             a[source] = 0
             empty_locs[i] = source
 
             # check that the number of empty cells is unchanged
             num_empty2 = NP.sum(a==0)
             assert num_empty == num_empty2
 
         # return the average fraction of similar neighbors
         return NP.nanmean(frac_same)
 
     def draw(self):
         """Draws the cells."""
         return draw_array(self.array, cmap=cmap, vmax=2)
 
 #SUGARSCAPE (artificial society where at every step the agents look around, find the closest agent filled with sugar, move and metabolize, they also become old, reproduce or die)

 def make_locs(n, m):
     """Makes array where each row is an index in an `n` by `m` grid.
 
     n: int number of rows
     m: int number of cols
 
     returns: NumPy array
     """
     t = [(i, j) for i in range(n) for j in range(m)]
     return NP.array(t)
 
 def make_visible_locs(vision):
     """Computes the kernel of visible cells.
 
     vision: int distance
     """
     def make_array(d):
         """Generates visible cells with increasing distance."""
         a = NP.array([[-d, 0], [d, 0], [0, -d], [0, d]])
         NP.random.shuffle(a)
         return a
 
     arrays = [make_array(d) for d in range(1, vision+1)]
     return NP.vstack(arrays)
 
 def distances_from(n, i, j):
     """Computes an array of distances.
 
     n: size of the array
     i, j: coordinates to find distance from
 
     returns: array of float
     """
     X, Y = NP.indices((n, n))
     return NP.hypot(X-i, Y-j)
 
 
 class Sugarscape(Cell2D):
     """Represents an Epstein-Axtell Sugarscape."""
 
     def __init__(self, n, **params):
         """Initializes the attributes.
 
         n: number of rows and columns
         params: dictionary of parameters
         """
         self.n = n
         self.params = params
 
         # track variables
         self.agent_count_seq = []
 
         # make the capacity array
         self.capacity = self.make_capacity()
 
         # initially all cells are at capacity
         self.array = self.capacity.copy()
 
         # make the agents
         self.make_agents()
 
     def make_capacity(self):
         """Makes the capacity array."""
 
         # compute the distance of each cell from the peaks.
         dist1 = distances_from(self.n, 15, 15)
         dist2 = distances_from(self.n, 35, 35)
         dist = NP.minimum(dist1, dist2)
 
         # cells in the capacity array are set according to dist from peak
         bins = [21, 16, 11, 6]
         a = NP.digitize(dist, bins)
         return a
 
     def make_agents(self):
         """Makes the agents."""
 
         # determine where the agents start and generate locations
         n, m = self.params.get('starting_box', self.array.shape)
         locs = make_locs(n, m)
         NP.random.shuffle(locs)
 
         # make the agents
         num_agents = self.params.get('num_agents', 400)
         assert(num_agents <= len(locs))
         self.agents = [Agent(locs[i], self.params)
                        for i in range(num_agents)]
 
         # keep track of which cells are occupied
         self.occupied = set(agent.loc for agent in self.agents)
 
     def grow(self):
         """Adds sugar to all cells and caps them by capacity."""
         grow_rate = self.params.get('grow_rate', 1)
         self.array = NP.minimum(self.array + grow_rate, self.capacity)
 
     def look_and_move(self, center, vision):
         """Finds the visible cell with the most sugar.
 
         center: tuple, coordinates of the center cell
         vision: int, maximum visible distance
 
         returns: tuple, coordinates of best cell
         """
         # find all visible cells
         locs = make_visible_locs(vision)
         locs = (locs + center) % self.n
 
         # convert rows of the array to tuples
         locs = [tuple(loc) for loc in locs]
 
         # select unoccupied cells
         empty_locs = [loc for loc in locs if loc not in self.occupied]
 
         # if all visible cells are occupied, stay put
         if len(empty_locs) == 0:
             return center
 
         # look up the sugar level in each cell
         t = [self.array[loc] for loc in empty_locs]
 
         # find the best one and return it
         # (in case of tie, argmax returns the first, which
         # is the closest)
         i = NP.argmax(t)
         return empty_locs[i]
 
     def harvest(self, loc):
         """Removes and returns the sugar from `loc`.
 
         loc: tuple coordinates
         """
         sugar = self.array[loc]
         self.array[loc] = 0
         return sugar
 
     def step(self):
         """Executes one time step."""
         replace = self.params.get('replace', False)
 
         # loop through the agents in random order
         random_order = NP.random.permutation(self.agents)
         for agent in random_order:
 
             # mark the current cell unoccupied
             self.occupied.remove(agent.loc)
 
             # execute one step
             agent.step(self)
 
             # if the agent is dead, remove from the list
             if agent.is_starving() or agent.is_old():
                 self.agents.remove(agent)
                 if replace:
                     self.add_agent()
             else:
                 # otherwise mark its cell occupied
                 self.occupied.add(agent.loc)
 
         # update the time series
         self.agent_count_seq.append(len(self.agents))
 
         # grow back some sugar
         self.grow()
         return len(self.agents)
 
     def add_agent(self):
         """Generates a new random agent.
 
         returns: new Agent
         """
         new_agent = Agent(self.random_loc(), self.params)
         self.agents.append(new_agent)
         self.occupied.add(new_agent.loc)
         return new_agent
 
     def random_loc(self):
         """Choose a random unoccupied cell.
 
         returns: tuple coordinates
         """
         while True:
             loc = tuple(NP.random.randint(self.n, size=2))
             if loc not in self.occupied:
                 return loc
 
     def draw(self):
         """Draws the cells."""
         draw_array(self.array, cmap='YlOrRd', vmax=9, origin='lower')
 
         # draw the agents
         xs, ys = self.get_coords()
         self.points = plt.plot(xs, ys, '.', color='red')[0]
 
     def get_coords(self):
         """Gets the coordinates of the agents.
 
         Transforms from (row, col) to (x, y).
 
         returns: tuple of sequences, (xs, ys)
         """
         agents = self.agents
         rows, cols = NP.transpose([agent.loc for agent in agents])
         xs = cols + 0.5
         ys = rows + 0.5
         return xs, ys
 
  
 class Agent:
 
     def __init__(self, loc, params):
         """Creates a new agent at the given location.
 
         loc: tuple coordinates
         params: dictionary of parameters
         """
         self.loc = tuple(loc)
         self.age = 0
         self.params = params  # Store input dictionary for later reference
 
         # extract the parameters
         max_vision = params.get('max_vision', 6) #How many km away an agent can see to search for resources.
         max_metabolism = params.get('max_metabolism', 4) #The amount of sugar an agent must consume per time step to stay alive.
         min_lifespan = params.get('min_lifespan', 1)
         max_lifespan = params.get('max_lifespan', 9)
         min_sugar = params.get('min_sugar', 0)
         max_sugar = params.get('max_sugar', 100) #The total amount of accumulated sugar an agent has harvested
 
         # choose attributes
         self.vision = NP.random.randint(1, max_vision+1)
         self.metabolism = NP.random.uniform(1, max_metabolism)
         self.lifespan = NP.random.uniform(min_lifespan, max_lifespan)
         self.sugar = NP.random.uniform(min_sugar, max_sugar)
 
     def step(self, env):
         """Look around, move, and harvest.
 
         env: Sugarscape
         """
         self.loc = env.look_and_move(self.loc, self.vision)
         self.sugar += env.harvest(self.loc) - self.metabolism
         self.age += 1
 
     def is_starving(self):
         """Checks if sugar has gone negative."""
         return self.sugar < 0
 
     def is_old(self):
         """Checks if lifespan is exceeded."""
         return self.age > self.lifespan
 

 #Boltzman wealth distribution (simulate a network of people starting with 1 unit of wealth that During each time-step, each agents with positive wealth randomly selects a trading partner and gives them one unit of their wealth)

 
 class WealthAgent(AP.Agent):
  def setup(self):
      self.wealth = 1
  def wealth_transfer(self):
   if self.wealth > 0:
    partner = self.model.agents.random()
    partner.wealth += 1
    self.wealth -= 1
 
 def gini(x):
  x = NP.array(x)
  mad = NP.abs(NP.subtract.outer(x, x)).mean()  # Mean absolute difference
  rmad = mad / NP.mean(x)  # Relative mean absolute difference
  return 0.5 * rmad
 
 class boltzman(AP.Model):
  def setup(self):
   self.agents = AP.AgentList(self, self.p.agents, WealthAgent)
  def step(self):
   self.agents.wealth_transfer()
  def update(self):
   self.record('Gini Coefficient', gini(self.agents.wealth))
  def end(self):
   self.agents.record('wealth')
 


 
 #SIR virus spread (people, which can be in one of the following three conditions: susceptible to the disease (S), infected (I), or recovered (R). The agents are connected to each other through a small-world network of peers. At every time-step, infected agents can infect their peers or recover from the disease based on random chance.)
 
 class Person(AP.Agent):
     
     def setup(self):
         """ Initialize a new variable at agent creation. """
         self.condition = 0  # Susceptible = 0, Infected = 1, Recovered = 2
 
     def being_sick(self):
         """ Spread disease to peers in the network. """
         rng = self.model.random
         for n in self.network.neighbors(self):
             if n.condition == 0 and self.p.infection_chance > rng.random():
                 n.condition = 1  # Infect susceptible peer
         if self.p.recovery_chance > rng.random():
             self.condition = 2  # Recover from infection
 
 
 class VirusModel(AP.Model):
 
     def setup(self):
         """ Initialize the agents and network of the model. """
 
         # Prepare a small-world network
         graph = NX.watts_strogatz_graph(
             self.p.population,
             self.p.number_of_neighbors,
             self.p.network_randomness)
 
         # Create agents and network
         self.agents = AP.AgentList(self, self.p.population, Person)
         self.network = self.agents.network = AP.Network(self, graph)
         self.network.add_agents(self.agents, self.network.nodes)
 
         # Infect a random share of the population
         I0 = int(self.p.initial_infection_share * self.p.population)
         self.agents.random(I0).condition = 1
 
     def update(self):
         """ Record variables after setup and each step. """
 
         # Record share of agents with each condition
         for i, c in enumerate(('S', 'I', 'R')):
             n_agents = len(self.agents.select(self.agents.condition == i))
             self[c] = n_agents / self.p.population
             self.record(c)
 
         # Stop simulation if disease is gone OR if user-defined max steps reached
         if self.I == 0 or (hasattr(self.p, 'steps') and self.t >= self.p.steps):
             self.stop()
 
     def step(self):
         """ Define the model's events per simulation step. """
 
         # Call 'being_sick' for infected agents
         self.agents.select(self.agents.condition == 1).being_sick()
 
     def end(self):
         """ Record evaluation measures at the end of the simulation. """
 
         # Record final evaluation measures
         self.report('Total share infected', self.I + self.R)
         self.report('Peak share infected', max(self.log['I']))
 


 #TURCHIN Multi-Path Forecasting: a contagion algorithm (variant of the SIR model with  Naive, Radicalized and Moderate individuals, where naive can be infected by radiczlized, and moderate correspond to the recovered ones); a political stress index prediction model (the growing inverse relative income, an average population age around 23 years, and the growth of the elite with respect to the population, increase the probability of radicalization contagion); and an elite dynamics module (if the elites do not differ in their relative income from commoners, the rate of relative elite on the whole populaiton is simply the net rate of social mobility, otherwise the surplus elites, those for whom elite positions are not available, tend to become radicalized, except for those who became moderates)
 
 mpfdoc='''
 Explanation of the Code:
 
 Individual Class:
 
 Inherits from AP.Agent.
 condition: Represents the state of the individual (0: Naive, 1: Radicalized, 2: Moderate).
 relative_income: Stores the individual's relative income.
 is_elite: A boolean indicating if the individual is part of the elite.
 contagion(): Implements the radicalization contagion. A naive individual can become radicalized if they have a radicalized neighbor and the political_stress_index is high enough (compared to a random number).
 recover(): Implements the transition from radicalized to moderate based on a moderation_chance.
 MPFModel Class:
 
 Inherits from AP.Model.
 setup():
 Initializes the population size and elite share.
 Creates an AP.AgentList of Individual agents.
 Creates a barabasi network using networkx and assigns it to the agents.
 Randomly selects a fraction of the population to be the initial elite.
 Sets the initial relative income for all individuals.
 Initializes a small fraction of the population as initially radicalized.
 Sets the equilibrium_elite_share.
 Initializes the political_stress_index.
 calculate_political_stress():
 Calculates the average relative income of the population.
 Computes the immiseration factor (inverse of relative income).
 Calculates elite_overproduction (current elite share minus the equilibrium share).
 Implements a simplified youth_bulge_factor based on the average_age parameter.
 Calculates the political_stress_index using Turchin's formula: α(t)=α 0 +α w (w 0 –w)+α e (e–e 0 )+A 20
   where:
 α0 =0.1 (base level)
 αw =1 (weight for immiseration)
 αe =0.5 (weight for elite overproduction)
 w0 is the initial_relative_income
 w is the current average relative income
 e is the current elite share
 e0 is the equilibrium_elite_share
 A20 is the contribution of the youth bulge.
 Ensures the political_stress_index is not negative.
 update_elite_dynamics():
 Calculates the average income for commoners and elites.
 Determines if there's a significant relative income difference between elites and commoners.
 Implements a simplified model of social mobility based on net_social_mobility_rate.
 If elites do not have a significantly higher relative income, the number of elites adjusts towards an equilibrium based on social mobility.
 If there's a surplus of elites (more than the equilibrium), a fraction of them (those who are not already radicalized) may become radicalized or moderate.
 Adjusts the number of elite individuals in the population to move towards the target elite count.
 update_relative_income():
 Updates the relative income of all agents based on a provided income_trend function. This is a simplified implementation; a more complex economic model could be integrated here.
 step():
 Calls update_relative_income() to update the economic driver.
 Calls calculate_political_stress() to update the environment.
 Calls contagion() on all agents to simulate the spread of radicalization.
 Calls recover() on all agents to simulate the transition to the moderate state.
 Calls update_elite_dynamics() to adjust the elite population.
 Records the proportion of Naive, Radicalized, and Moderate individuals, the political_stress_index, the average relative income, and the elite share.
 end():
 Records the final and peak shares of radicalized individuals.
 income_trajectory(t) Function:
 
 A sample function that defines how the average relative income changes over time (t). You can customize this function to simulate different economic scenarios.
 Parameter Dictionary:
 
 parameters: A dictionary holding all the model parameters, such as population size, initial conditions, network properties, and parameters for the political stress and elite dynamics.
 Model Instantiation and Running:
 
 An instance of MPFModel is created with the defined parameters.
 model.run() executes the simulation.
 '''
 
 class Individual(AP.Agent):
     """ An individual in the Multi-Path Forecasting model. """
 
     def setup(self):
         """ Initialize individual attributes. """
         self.condition = 0  # 0: Naive, 1: Radicalized, 2: Moderate (Recovered)
         self.relative_income = self.model.p.initial_relative_income
         self.is_elite = False
 
     def contagion(self):
         """ Contagion of radicalization based on neighbors' conditions. """
         if self.condition == 0:  # Naive individual
             rng = self.model.random
             for neighbor in self.network.neighbors(self):
                 if neighbor.condition == 1 and self.model.political_stress_index > rng.random():
                     self.condition = 1  # Become radicalized
 
     def recover(self):
         """ Individuals can become moderate (recovered). """
         if self.condition == 1:  # Radicalized individual
             rng = self.model.random
             if self.model.p.moderation_chance > rng.random():
                 self.condition = 2
 
 class MPFModel(AP.Model):
     """ An Agent-Based Model implementing Turchin's Multi-Path Forecasting. """
 
     def setup(self):
         """ Initialize the population, network, and initial conditions. """
         self.population_size = self.p.population
         self.initial_elite_share = self.p.initial_elite_share
         self.agents = AP.AgentList(self, self.population_size, Individual)
 
         # Create a small-world network
         #graph = NX.watts_strogatz_graph(
         graph = NX.barabasi_albert_graph(
             self.population_size,
             self.p.number_of_neighbors,
             #self.p.network_randomness
         )
         self.network = self.agents.network = AP.Network(self, graph)
         self.network.add_agents(self.agents, self.network.nodes)
 
         # Initialize elite individuals
         initial_elite_count = int(self.initial_elite_share * self.population_size)
         elite_indices = self.random.sample(range(self.population_size), initial_elite_count)
         for i in elite_indices:
             self.agents[i].is_elite = True
 
         # Initialize relative income for all individuals
         for agent in self.agents:
             agent.relative_income = self.p.initial_relative_income
 
         # Initialize a small fraction of the population as initially radicalized
         initial_radicalized_count = int(self.p.initial_radicalized_share * self.population_size)
         radicalized_indices = self.random.sample(range(self.population_size), initial_radicalized_count)
         for i in radicalized_indices:
             self.agents[i].condition = 1
 
         # Set initial equilibrium elite share
         self.equilibrium_elite_share = 0.01
 
         # Initialize political stress index
         self.political_stress_index = 0.0
 
     def calculate_political_stress(self):
         """ Calculate the political stress index based on immiseration and elite overproduction. """
         # Immiseration (inverse relative income)
         average_relative_income = NP.mean([agent.relative_income for agent in self.agents])
         immiseration = 1 / average_relative_income if average_relative_income > 0 else 0
 
         # Elite overproduction
         current_elite_share = sum(1 for agent in self.agents if agent.is_elite) / self.population_size
         elite_overproduction = current_elite_share - self.equilibrium_elite_share
 
         # Age structure (simplified - assuming a constant average age for now)
         youth_bulge_factor = 1 if 20 <= self.p.average_age <= 26 else 0  # Simplified youth bulge
 
         alpha_0 = 0.1
         alpha_w = 1
         alpha_e = 0.5
         A_20 = youth_bulge_factor * self.p.youth_bulge_weight  # Contribution of youth bulge
 
         self.political_stress_index = alpha_0 + alpha_w * (self.p.initial_relative_income - average_relative_income) + alpha_e * elite_overproduction + A_20
         self.political_stress_index = max(0, self.political_stress_index) # Ensure it's not negative
 
     def update_elite_dynamics(self):
         """ Update the elite population based on relative income differences and social mobility. """
         average_commoner_income = NP.mean([agent.relative_income for agent in self.agents if not agent.is_elite]) if any(not agent.is_elite for agent in self.agents) else self.p.initial_relative_income
         average_elite_income = NP.mean([agent.relative_income for agent in self.agents if agent.is_elite]) if any(agent.is_elite for agent in self.agents) else self.p.initial_relative_income
 
         relative_income_difference = average_elite_income - average_commoner_income
 
         # Net rate of social mobility (simplified for now)
         net_social_mobility_rate = self.p.social_mobility_rate
 
         current_elite_count = sum(1 for agent in self.agents if agent.is_elite)
 
         if abs(relative_income_difference) < 1e-6:  # Elites do not differ in relative income
             target_elite_count = int((self.equilibrium_elite_share + net_social_mobility_rate) * self.population_size)
         else:
             # Surplus elites tend to become radicalized or moderate
             surplus_elites = current_elite_count - int(self.equilibrium_elite_share * self.population_size)
             if surplus_elites > 0:
                 non_elite_agents = [agent for agent in self.agents if not agent.is_elite and agent.condition != 1] # Avoid radicalizing already radicalized
                 if non_elite_agents:
                     num_to_radicalize = min(surplus_elites, len(non_elite_agents))
                     radicalize_candidates = self.random.sample(non_elite_agents, num_to_radicalize)
                     for agent in radicalize_candidates:
                         if self.random.random() > self.p.moderation_chance: # Some become moderate
                             agent.condition = 1 # Become radicalized
                         else:
                             agent.condition = 2 # Become moderate
 
             target_elite_count = int((self.equilibrium_elite_share + net_social_mobility_rate) * self.population_size)
 
         # Adjust the number of elites towards the target
         diff_elite = target_elite_count - current_elite_count
         if diff_elite > 0:
             non_elite_candidates = [agent for agent in self.agents if not agent.is_elite]
             if non_elite_candidates:
                 promote_indices = self.random.sample(range(len(non_elite_candidates)), min(diff_elite, len(non_elite_candidates)))
                 for index in promote_indices:
                     non_elite_candidates[index].is_elite = True
         elif diff_elite < 0:
             elite_candidates = [agent for agent in self.agents if agent.is_elite]
             if elite_candidates:
                 demote_indices = self.random.sample(range(len(elite_candidates)), min(-diff_elite, len(elite_candidates)))
                 for index in demote_indices:
                     elite_candidates[index].is_elite = False
 
     def update_relative_income(self):
         """ Update the relative income distribution (simplified). """
         # This is a very simplified implementation. A more detailed model would be needed.
         w_min = 0.15
         w_max = 0.95
         self.current_relative_income = max(w_min, min(w_max, self.p.income_trend(self.t)))
         for agent in self.agents:
             agent.relative_income = self.current_relative_income
 
     def step(self):
         """ Define the model's events per step. """
         self.update_relative_income()
         self.calculate_political_stress()
         self.agents.contagion()
         self.agents.recover()
         self.update_elite_dynamics()
 
         # Record variables
         self.record("Naive", len(self.agents.select(self.agents.condition == 0)) / self.population_size)
         self.record("Radicalized", len(self.agents.select(self.agents.condition == 1)) / self.population_size)
         self.record("Moderate", len(self.agents.select(self.agents.condition == 2)) / self.population_size)
         self.record("PSI", self.political_stress_index)
         #self.record("avg Income", NP.mean([agent.relative_income for agent in self.agents]))
         #self.record("Elite Share", sum(1 for agent in self.agents if agent.is_elite) / self.population_size)
 
     def end(self):
         """ Record evaluation measures at the end of the simulation. """
         self.report("Final Radicalized Share", self.log["Radicalized"][-1])
         self.report("Peak Radicalized Share", max(self.log["Radicalized"]))
 
 def income_trajectory(t):
     """ A sample function for the trajectory of relative income. """
     # Example: A sinusoidal trend with a general downward tendency
     amplitude = 0.05
     frequency = 0.02
     downward_trend = -0.0005 * t
     initial_w = 0.9
     return initial_w + amplitude * NP.sin(frequency * t) + downward_trend
 
 
 
 if '-a.mf=' in o: # run Turchin's MPF
  r_=re.findall(r'-a.mf=(.+?) ',o); r_=r_[0].split(',');
  if 'x_' in locals():
   steps=len(x_)
  else: steps=10 #default steps
  print(f"running MultiPath Forecasting algorithm (https://en.wikipedia.org/wiki/Cliodynamics)")
  

  parameters = {
      'population': int(NP.log2(int(r_[0]))*100),
      'initial_relative_income': 0.9, 
      'initial_elite_share': 0.02, #2% population is elite
      'initial_radicalized_share': 0.01,
      'number_of_neighbors': 2,
      'network_randomness': 0.2,
      'moderation_chance': 0.05,
      'social_mobility_rate': float(r_[1]),
      'average_age': 23,
      'youth_bulge_weight': 0.1, #10% pop is youth
      'income_trend': income_trajectory,
      'steps': int(steps)
  }
  
  # Run the MPF model
  mpf_model = MPFModel(parameters)
  results = mpf_model.run()
  print(results.variables.MPFModel)
  results=results.variables.MPFModel
  print('\nevaluation')
  results=PD.concat([results, x_], axis=1); print(results.corr())
  print('---END PROCESS---'); sys.exit();
 
 if '-a.wd=' in o: #boltzman wealth distribution
  r_=re.findall(r'-a.wd=(.+?) ',o); r_=r_[0].split(',');
  if 'x_' in locals():
   steps=len(x_)
  else: steps=10 #default steps

  print(f"running Boltzman wealth distribution algorithm (https://en.wikipedia.org/wiki/Boltzmann_Fair_Division)")

  parameters = {'agents': int(NP.log2(int(r_[0]))*100), 'steps': steps}
  model = boltzman(parameters)
  results = model.run()
  data = results.variables.boltzman
  print(data)
  print('\nevaluation')
  results=PD.concat([data, x_], axis=1); print(results.corr())
  print('---END PROCESS---'); sys.exit();
 
 if '-a.sir=' in o: #sir virus spread model
  r_=re.findall(r'-a.sir=(.+?) ',o); r_=r_[0].split(',');
  if 'x_' in locals():
   steps=len(x_)
  else: steps=100 #default steps

  print(f"running SIR algorithm (https://en.wikipedia.org/wiki/Mathematical_modelling_of_infectious_diseases#The_SIR_model)")

  parameters = {
      'population': int(NP.log2(int(r_[0]))*100),
      'infection_chance': float(r_[1]),
      'recovery_chance': 0.03,
      'initial_infection_share': 0.1,
      'number_of_neighbors': 2,
      'network_randomness': 0.5,
      'steps': steps,
  }
  
  model = VirusModel(parameters)
  results = model.run()
  print(results.variables.VirusModel)
  results=results.variables.VirusModel
  print('\nevaluation')
  results=PD.concat([results, x_], axis=1); print(results.corr())
  print('---END PROCESS---'); sys.exit();



 if '-a.ss=' in o: #shelling segregation
  r_=re.findall(r'-a.ss=(.+?) ',o); r_=r_[0].split(',');
  if 'x_' in locals():
   steps=len(x_)
  else: steps=10 #default steps
  print(f"running Shelling segregation algorithm (https://en.wikipedia.org/wiki/Schelling%27s_model_of_segregation)")

  grid = Schelling(n=int(r_[0]), p=float(r_[1]))  # Adjust grid size and threshold as needed
  # Collect data as a list of dictionaries
  segregation_scores = []
  for i in range(steps):
      score = grid.step()
      segregation_scores.append({ "Segregation Score": score})

  df = PD.DataFrame(segregation_scores)
  print(df)
  print('\nevaluation')
  results=PD.concat([df, x_], axis=1); print(results.corr())
  print('---END PROCESS---'); sys.exit();
 
 # You can now visualize the segregation scores:
 #plt.plot(range(steps), segregation_scores)
 #plt.xlabel("Step")
 #plt.ylabel("Segregation Score")
 #plt.title("Schelling Segregation Simulation")
 #plt.show()
 
 
 
 if '-a.s=' in o:  #epstein & axtell's sugarscape
  r_=re.findall(r'-a.s=(.+?) ',o); r_=r_[0].split(',');
  if 'x_' in locals():
   steps=len(x_)
  else: steps=10 #default steps

  print(f"running sugarscape algorithm (https://en.wikipedia.org/wiki/Sugarscape)")
  env = Sugarscape(int(NP.log2(int(r_[0]))*100),
                  num_agents=int(NP.log2(int(r_[1]))*100),
                  #min_lifespan=0,
                  #max_lifespan=3,
                  replace=True)
  #print(dir(env))
  #for step in range(10):  # Example: run for 10 steps
  #    num_agents_at_step = env.step()
  #    print(f"Step {step + 1}: Number of agents = {num_agents_at_step}")
  #
  #    for agent in env.agents:
  #        print(f"  Agent at ({agent.loc}): Vision={agent.vision}, Metabolism={agent.metabolism}, Sugar={agent.sugar}, Age={agent.age}")
  avg_scores=[]
  timesteps=steps
  for step in range(timesteps):  # Run the simulation for 50 steps
   visions = [agent.vision for agent in env.agents]
   metabolisms = [agent.metabolism for agent in env.agents]
   sugars = [agent.sugar for agent in env.agents]
   ages = [agent.age for agent in env.agents]
   avg_scores.append({"agents": env.step(), "avgVision": NP.mean(visions), "avgMetabolism": NP.mean(metabolisms), "avgSugar": NP.mean(sugars), "avgAge": NP.mean(ages)})
 
  df = PD.DataFrame(avg_scores)
  dfn = (df - df.min()) / (df.max() - df.min())
  print(dfn)
  print('\nevaluation')
  results=PD.concat([dfn, x_], axis=1); print(results.corr())
  print('---END PROCESS---'); sys.exit();




#---process mining

if '-m.pnam' in o: # *petri net from alpha miner algorithm*
 k=10
 if '-m.pnam=' in o:
  r_=re.findall(r'-m.pnam=(.+?) ', o); k=int(r_[0]);
 print('Extract Petri Net with Alpha Miner https://en.wikipedia.org/wiki/Petri_net \nProcess mining data must contain only 3 columns: timestamp,activity,caseid.\ntheory: https://en.wikipedia.org/wiki/Process_mining');
 os.system('pip install -U -q pm4py')
 import pm4py
 import pandas as pd
 from pm4py.objects.conversion.log import converter as log_converter
 from pm4py.objects.log.util import dataframe_utils
 x_ = x_.rename(columns={'timestamp': 'time:timestamp', 'activity': 'concept:name', 'caseid': 'case:concept:name'}) # Rename columns to match PM4Py's expected names
 x_['time:timestamp'] = PD.to_datetime(x_['time:timestamp']) # Ensure the timestamp column is in datetime format
 x_ = x_.sort_values('time:timestamp') # Sort the DataFrame by timestamp
 log = log_converter.apply(x_)
 #filter_variants_by_coverage_percentage
 log=pm4py.filter_variants_top_k(log, k, activity_key='concept:name', timestamp_key='time:timestamp',  case_id_key='case:concept:name'); #print(f"log={log}")
 net, im, fm = pm4py.discover_petri_net_alpha(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp'); 
 pm4py.view_petri_net(net, im, fm, rankdir='TB'); 
 conform = pm4py.conformance.conformance_diagnostics_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp');
 rf=0
 for i in conform:
  rf=rf+i['fitness']
 rf=round(rf/len(conform),3)
 print(f"replay fitness: {rf}")
 print('---END PROCESS---'); sys.exit();

if '-m.hnas' in o: # heuristic miner
 k=10
 if '-m.hnas=' in o:
  r_=re.findall(r'-m.hnas=(.+?) ', o); k=int(r_[0]);
 print('Extract Heuristic net from A* pathfinding algorithm https://en.wikipedia.org/wiki/A*_search_algorithm \nProcess mining data must contain only 3 columns: timestamp,activity,caseid.\ntheory: https://en.wikipedia.org/wiki/Process_mining');
 os.system('pip install -U -q pm4py')
 import pm4py
 import pandas as pd
 from pm4py.objects.conversion.log import converter as log_converter
 from pm4py.objects.log.util import dataframe_utils
 x_ = x_.rename(columns={'timestamp': 'time:timestamp', 'activity': 'concept:name', 'caseid': 'case:concept:name'}) # Rename columns to match PM4Py's expected names
 x_['time:timestamp'] = PD.to_datetime(x_['time:timestamp']) # Ensure the timestamp column is in datetime format
 x_ = x_.sort_values('time:timestamp') # Sort the DataFrame by timestamp
 log = log_converter.apply(x_)
 #filter_variants_by_coverage_percentage
 log=pm4py.filter_variants_top_k(log, k, activity_key='concept:name', timestamp_key='time:timestamp',  case_id_key='case:concept:name'); #print(f"log={log}")
 net = pm4py.discover_heuristics_net(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp'); 
 pm4py.view_heuristics_net(net)
 conform = pm4py.conformance.conformance_diagnostics_alignments(log, net, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp');
 rf=0
 for i in conform:
  rf=rf+i['fitness']
 rf=round(rf/len(conform),3)
 print(f"replay fitness: {rf}")
 print('---END PROCESS---'); sys.exit();

if '-m.bpmn' in o: # bpmn inductive miner
 k=10
 if '-m.bpmn=' in o:
  r_=re.findall(r'-m.bpmn=(.+?) ', o); k=int(r_[0]);
 print('Extract BPMN using inductive miner algorithm https://en.wikipedia.org/wiki/Business_Process_Model_and_Notation \nProcess mining data must contain only 3 columns: timestamp,activity,caseid.\ntheory: https://en.wikipedia.org/wiki/Process_mining');
 os.system('pip install -U -q pm4py')
 import pm4py
 import pandas as pd
 from pm4py.objects.conversion.log import converter as log_converter
 from pm4py.objects.log.util import dataframe_utils
 x_ = x_.rename(columns={'timestamp': 'time:timestamp', 'activity': 'concept:name', 'caseid': 'case:concept:name'}) # Rename columns to match PM4Py's expected names
 x_['time:timestamp'] = PD.to_datetime(x_['time:timestamp']) # Ensure the timestamp column is in datetime format
 x_ = x_.sort_values('time:timestamp') # Sort the DataFrame by timestamp
 log = log_converter.apply(x_)
 log=pm4py.filter_variants_top_k(log, k, activity_key='concept:name', timestamp_key='time:timestamp',  case_id_key='case:concept:name'); #print(f"log={log}")
 net = pm4py.discover_bpmn_inductive(log); 
 pm4py.view_bpmn(net, rankdir='TB')
 conform = pm4py.conformance.conformance_diagnostics_alignments(log, net, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp');
 rf=0
 for i in conform:
  rf=rf+i['fitness']
 rf=round(rf/len(conform),3)
 print(f"replay fitness: {rf}")
 print('---END PROCESS---'); sys.exit();



if '-u.arl' in o: #association rule mining
 xn_=x_.to_numpy(); xn_=xn_.flatten(); xn_=NP.array(xn_, dtype=str)
 print('apply association rule learning (market basket analysis).\ntheory: https://en.wikipedia.org/wiki/Association_rule_learning\ndocs: https://github.com/rasbt/mlxtend');
 xn_=NP.char.split(xn_,sep=' '); 
 from mlxtend.preprocessing import TransactionEncoder;
 import mlxtend;
 from mlxtend.frequent_patterns import apriori, association_rules;
 te = TransactionEncoder()
 te_ary = te.fit(xn_).transform(xn_)
 df = PD.DataFrame(te_ary, columns=te.columns_); 
 frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True); 
 #frequent_itemsets = fpmax(df, min_support=0.01, use_colnames=True)
 results=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1);
 af= open('log.txt', 'a'); af.write(results.to_string()+"\n\n"); af.close(); 
 print(results);
 print('results printed in log.txt file'); timestamp=DT.datetime.now(); 
 print(f"-u.arl stops other tasks\ntime:{timestamp}");
 inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();


#---automatically detect target class type
if task=='s': 
 if y_.dtype.kind in 'biufc' and not '-d.t=c' in o:
  #y_=(y_-y_.min())/(y_.max()-y_.min()); 
  y_=y_.astype('float'); target='r'; 
  print('read target as number, turned to float. to read it as a label use -d.t=c');
  if '-p.cn' in o:
   y_=(y_-y_.min())/(y_.max()-y_.min()); 
   print('apply target numbers 0-1 normalization');
 else:
  y_=y_.astype('category').cat.codes.astype('int'); 
  target='c'; print('read target as label, turned to integer category')


#---manage filling missing values in features
if 'x_' in locals() and not '-u.arl' in o:
 if x_.isnull().values.any():
  x_ = x_.fillna(0); print('filling missing values with 0 by default'); 

if 'y_' in locals() and '-d.m=1' in o: #replace missing values with mode or mean
 if y_.isnull().values.any():
  if target=='r':
   y_ = y_.fillna(y_.mean()); 
   print('WARNING: missing values in the target class, filled with the mean.'); 
  if target=='c':
   y_ = y_.fillna(y_.mode()); 
   print('WARNING: missing values in the target class, filled with the mode.'); 


if 't_' in locals():
 t_=t_.astype(str);


#---inter annotator ageement
if '-u.irr' in o:
 print(f'compute Inter-rater Reliability between {x_.columns}. All columns must be numerical.');
 print('theory: https://en.wikipedia.org/wiki/Inter-rater_reliability');
 t= x_.values.tolist()
 z=[]
 ts=NP.array(t);
 tss=ts.shape;
 v=len(NP.unique(ts)) +1 ; print(f"values {v}");
 r=tss[1]; print(f"raters {r}");
 e=tss[0]; print(f"examples {e}");
 for i in t:
  y=[];
  for x in range(v):
   y.append(i.count(x))
  z.append(y)

 ra = NP.array(t)
 t0=ra[:, 0]
 t1=ra[:, 1]
 from sklearn.metrics import cohen_kappa_score
 c=cohen_kappa_score(t1, t0)
 from statsmodels.stats.inter_rater import fleiss_kappa
 k=fleiss_kappa(z)
 os.system('pip install agreement -q')
 z = NP.array(z)
 from agreement.metrics import scotts_pi, krippendorffs_alpha
 pi = scotts_pi(z)
 alpha = krippendorffs_alpha(z)
 print(f'c={c}')
 print(f'k={k}')
 print(f'a={alpha}')
 print(f'p={pi}')
 print('---END PROCESS---');  sys.exit();

#---data preprocessing 
if '-p.ir' in o:
 x_=SK.utils.shuffle(x_, random_state=1); 
 print('apply instance position randomization'); #

#if '-p.stnd' in o: *TO REMOVE*
# x_scaled=SK.preprocessing.StandardScaler().fit_transform(x_.values); 
#x_=PD.DataFrame(x_scaled, columns=x_.columns); 
#print('apply feature standardization')

if '-p.tl' in o: 
 t_=t_.str.lower(); print(f'apply text to lowercase filter:\n{t_}'); 

if '-p.tc' in o:
 t_=t_.str.replace("[^a-zA-Z0-9]", ' ', regex=True); 
 t_=t_.str.replace(" {2,30}", ' ', regex=True); 
 print(f'text cleaning from nonword characters and multiple spaces:\n{t_}');

if '-p.trs' in o:
 t_ = t_.str.replace(r'\b.{1,3}\b ', ' ',regex=True)
 t_ = t_.str.replace(r'\s+', ' ',regex=True)
 print(f'apply text-regex strip, removing words from 1 to 3 characters:\n{t_}');

if '-p.tsw=' in o:
 r_=re.findall(r'-p.tsw=(.+?) ',o); swlist=r_[0];
 swlist = swlist.replace(',',' | '); print(swlist); #get list of stopwords
 t_ = t_.str.replace(swlist, ' ')
 #t_ = t_.str.replace(r'\s+', ' ')
 print(f'removing stopwords: {swlist}:\n{t_}');

ncols=len(x_._get_numeric_data().columns); cols=len(x_.columns); 


if '-p.lda' in o and 't_' in locals():
 from sklearn.feature_extraction.text import CountVectorizer
 from sklearn.decomposition import LatentDirichletAllocation
 print('apply topic modeling with Latent Dirichlet Allocation.');
 print('theory: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation');
 nwordspertopic=5 #default words per topic
 if '-p.lda=' in o:
  r_=re.findall(r'-p.lda=(.+?) ',o); nwordspertopic=int(r_[0])
 # create vocabulary doc-term matrix (X)
 vectorizer = CountVectorizer()
 vec_ = vectorizer.fit_transform(t_); #print(top);
 #apply LDA
 lda_model = LatentDirichletAllocation(n_components=1, random_state=2)
 tl_=[]; #ts='';
 for v in vec_:
  lda_model.fit(v)
  words = vectorizer.get_feature_names_out()
  for idx, topic in enumerate(lda_model.components_):
   top=[words[i] for i in topic.argsort()[-nwordspertopic:][::-1]]
   ts = ' '.join(top); tl_.append(ts);
 t_=PD.DataFrame(tl_); print(t_)
 #print('---END PROCESS---');  sys.exit();


#---embeddings
if '-u.dcs' in o and 't_' in locals(): 
 print('apply cosine similarity from word embeddings with paraphrase-multilingual-MiniLM-L12-v2 model.');
 print('theory: https://en.wikipedia.org/wiki/Word_embedding');
 from sentence_transformers import SentenceTransformer, util
 embedmodel = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
 # Create a DataFrame to store results
 similar_words = PD.DataFrame(columns=["class1", "class2", "cosine_similarity"])
 # Loop through each pair of documents
 for irow in t_:
  for jrow in t_:
   pair=f"{irow}\n{jrow}"
   iembeddings = embedmodel.encode(irow, convert_to_tensor=True)
   jembeddings = embedmodel.encode(jrow, convert_to_tensor=True)
   #Compute cosine similarities
   cosine_similarities = util.cos_sim(iembeddings, jembeddings)
   similarity = cosine_similarities[0].item()
   if similarity<1:
    print(f"{pair}\ncosine similirity={similarity}\n")

 print('---END PROCESS---'); sys.exit();



#---feature extraction
fx=0; #define flag for feature extraction

#one-hot (if data not empty, extract features from labels, apply SVD)
if not x_.empty and not 'zip' in datatype and cols >= ncols: 
 x_=PD.get_dummies(x_); x_=x_.reset_index(drop=True); 
 if '-d.data' in o:
  print('async sparse one-hot matrix from labels:\n',x_)  
 else:
  print('apply one-hot binarization of labels. sparse matrix'); 
#SVD feature reduction
# if not '-d.r=0' in o and not '-u.cor' in o: 
#  if len(x_.columns) >2:
#   svd=SK.decomposition.TruncatedSVD(svdim, random_state=1); 
#   x_=PD.DataFrame(svd.fit_transform(x_)); 
#   if '-d.data' in o:
#    print('sync dense SVD matrix from one-hot labels:\n',x_)  
#   else: 
#    print('apply Singular Value Decomposition. obtain dense matrix\ntheory: https://en.wikipedia.org/wiki/Singular_value_decomposition');
#   print('theory: https://en.wikipedia.org/wiki/Singular_value_decomposition');
#  else:
#   x_=x_; print('tabular data is small, SVD not applied');


if 't_' in locals() and '-x.' in o: #extract features from text, apply LSA
 print('original text data:\n',t_) if '-d.data' in o else 0;

 if '-x.tm=' in o: #one hot token matrix
  orig_t_ = t_; #keep text for wordcloud
  r_=re.findall(r'-x.tm=(.+?) ', o); wu=int(r_[0]); fx=1;
  t=TF.keras.preprocessing.text.Tokenizer(num_words=wu, lower=False, oov_token=wu)
  t.fit_on_texts(t_); seq=t.texts_to_sequences(t_); 
  wv_=t.sequences_to_matrix(seq);  t_=PD.DataFrame(wv_);
  print('word indexes:\n',t.index_word);
  if '-d.data' in o:
   print('token freq matrix:\n',t_);
  else:
   print(f'extracting {wu} features from text');

 if '-x.cm=' in o: #one hot char matrix
  orig_t_ = t_; #keep text for wordcloud
  r_=re.findall(r'-x.cm=(.+?) ', o); wu=int(r_[0]); fx=1;
  t=TF.keras.preprocessing.text.Tokenizer(num_words=wu, char_level=True, oov_token=wu)
  t.fit_on_texts(t_); seq=t.texts_to_sequences(t_); 
  wv_=t.sequences_to_matrix(seq);  
  t_=PD.DataFrame(wv_);
  print('word indexes:\n',t.index_word);
  if '-d.data' in o:
   print('charachters matrix:\n',t_);
  else:
   print(f'extracting {wu} features from text');

 if '-x.ts=' in o: #one hot sequence matrix
  orig_t_ = t_; #keep text for wordcloud
  x_=PD.DataFrame(); 
  print('tabular data dropped to prevent mixing tabular data and sequence text'); 
  r_=re.findall(r'-x.ts=(.+?) ', o); wu=int(r_[0]); fx=1;
  t = TF.keras.preprocessing.text.Tokenizer(num_words=1000, lower=True, oov_token=0); 
  t.fit_on_texts(t_);
  seq = t.texts_to_sequences(t_); 
  seq=TF.keras.preprocessing.sequence.pad_sequences(seq, maxlen=wu); 
  print('word indexes:\n',t.index_word); 
  t_=PD.DataFrame(seq);  #print(t_[0]);
  #t_=TF.one_hot(seq,wu); print('one hot seq:', t_);  
  #dshape=(t_.shape[1],t_.shape[2]); t_=t_.numpy(); 
  #ft_=[]; [ft_.append(i.flatten()) for i in t_]; t_=PD.DataFrame(ft_);
  if '-d.data' in o:
   print('token sequence matrix:\n',t_);
  else:
   print(f'extracting {wu} features from text');

 if '-x.ng=' in o:
  orig_t_ = t_; #keep text for wordcloud
  r_=re.findall(r'-x.ng=(.+?) ',o); 
  mi=int(r_[0][0]); ma=int(r_[0][1]); ty=(r_[0][2]); mo=(r_[0][3]); 
  if len(r_[0]) ==5:
   mxf=int(r_[0][4])*100;
  else:
   mxf=1000;
  if ty=='c' and mo=='f':
   w=SK.feature_extraction.text.CountVectorizer(ngram_range=(mi,ma),analyzer='char_wb',max_features=mxf); 
   wv_=w.fit_transform(t_); fx=1;
   fn_=[]; [fn_.append(i) for i in w.get_feature_names_out()]; 
   t_=PD.DataFrame(wv_.toarray(), columns=fn_); 
   if '-d.data' in o:
    print('sparse char ngram matrix:\n',t_) 
   else:
    print(f'extract {mi}-{ma} char ngram from text'); print(fn_);
  if ty=='w' and mo=='f':
   w=SK.feature_extraction.text.CountVectorizer(ngram_range=(mi,ma),analyzer='word',max_features=mxf); 
   wv_=w.fit_transform(t_); fx=1;
   fn_=[]; [fn_.append(i) for i in w.get_feature_names_out()]; 
   t_=PD.DataFrame(wv_.toarray(), columns=fn_); 
   if '-d.data' in o:
    print('sparse word ngram matrix:\n',t_) 
   else:
    print(f'extract {mi}-{ma} word ngram from text'); print(fn_);  
  if ty=='c' and mo=='t':
   w=SK.feature_extraction.text.TfidfVectorizer(ngram_range=(mi,ma),analyzer='char_wb',max_features=mxf); 
   wv_=w.fit_transform(t_); fx=1;
   fn_=[]; [fn_.append(i) for i in w.get_feature_names_out()]; 
   t_=PD.DataFrame(wv_.toarray(), columns=fn_); 
   if '-d.data' in o:
    print('async sparse char ngram matrix:\n',t_) 
   else:
    print(f'extract {mi}-{ma} char ngram from text'); print(fn_);
  if ty=='w' and mo=='t':
   w=SK.feature_extraction.text.TfidfVectorizer(ngram_range=(mi,ma),analyzer='word',max_features=mxf); 
   wv_=w.fit_transform(t_); fx=1;
   fn_=[]; [fn_.append(i) for i in w.get_feature_names_out()]; 
   t_=PD.DataFrame(wv_.toarray(), columns=fn_); 
   if '-d.data' in o:
    print('sparse word ngram matrix:\n',t_) 
   else:
    print(f'extract {mi}-{ma} word ngram from text'); print(fn_);

  #lsadim=int(len(fn_)/2);
#  if not '-d.r=0' in o:
#   svd=SK.decomposition.TruncatedSVD(lsadim, random_state=1); 
#   t_=PD.DataFrame(svd.fit_transform(t_));
#   if '-d.data' in o:
#    print('sync dense LSA ngram matrix:\n',t_) 
#   else:
#    print('apply LSA to ngrams by default, obtain dense sync matrix\ntheory: https://en.wikipedia.org/wiki/Latent_semantic_analysis');
#   print('theory: https://en.wikipedia.org/wiki/Latent_semantic_analysis');
#  if '-d.save ' in o:
#   print(f"WARNING: test set must contain more than {lsadim} instances for compatibility"); 



 if '-x.trans' in o: #models: https://huggingface.co/models?sort=downloads
  print(f'extracting features with transformer models'); 
  print('theory: https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)');
  import torch  
  fx=1; orig_t_ = t_;
  if '-x.trans=bl' in o:
   print(f'using BERT large uncased for English');
   from transformers import BertTokenizer,BertModel  
   tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
   transmodel = BertModel.from_pretrained("bert-large-uncased")
  elif '-x.trans=bix' in o:
   print(f'using BERT Italian xxl');
   from transformers import AutoModel, AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-xxl-cased')
   transmodel = AutoModel.from_pretrained('dbmdz/bert-base-italian-xxl-cased')
  elif '-x.trans=r' in o:
   print(f'using RoBERTa base');
   from transformers import RobertaTokenizer, TFRobertaModel
   tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
   transmodel = TFRobertaModel.from_pretrained('roberta-base')
  else:
   print(f'using BERT multilanguage uncased');
   from transformers import BertTokenizer,BertModel
   tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased') 
   transmodel = BertModel.from_pretrained("google-bert/bert-base-multilingual-uncased")
  df =NP.array([]);
  for i in tqdm(range(len(t_))):
   sentence=t_[i]; 
   tokens = tokenizer.encode(sentence, padding=True, truncation=True,max_length=50, add_special_tokens=True, return_tensors="pt")
   output = transmodel(tokens)
   hiddenstates, features = output[0], output[1]
   df=NP.append(df,features.detach().numpy());
  print(''); #format output after tqdm
  df=NP.reshape(df,(x_.shape[0],features.shape[1]));
  t_=PD.DataFrame(df)
  t_ = t_.reset_index(drop=True);
  if '-d.data' in o:
   print('sync dense bert matrix:\n',t_)  
  else:
   print(f"extracted {features.shape[1]} features");



 if '-x.zsl=' in o: 
  r_=re.findall(r'-x.zsl=(.+?) ',o); #print(r_)
  print(f'extracting features with zero shot mDeBERTa-v3-base-xnli-multilingual'); 
  print('theory: https://en.wikipedia.org/wiki/Zero-shot_learning');
  import torch; 
  from transformers import pipeline;
  fx=1; orig_t_ = t_;
  classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
  candidate_labels = r_[0].split(','); 
  print(f"using the following zero-shot labels: {candidate_labels}");
  df =f"label,confidence\n";
  for i in tqdm(range(len(t_))):
   sentence=t_[i]; 
   result=classifier(sentence, candidate_labels);
   indexwinner=result['scores'].index(max(result['scores'])); 
   confidence=max(result['scores']); 
   labelwinner=result['labels'][indexwinner];
   df =df+f"{labelwinner},{confidence}\n";
  df= StringIO(df);
  df = PD.read_csv(df, sep=",");
  t_=PD.get_dummies(df, dtype='int');
  t_ = t_.reset_index(drop=True);
  if '-d.data' in o:
   print('predicted feature matrix:\n',t_)  



 if '-x.d=' in o: #user defined lexical resources
  fx=1; 
  orig_t_ = t_; #keep text for wordcloud
  r_=re.findall(r'-x.d=(.+?) ', o); 
  l=r_[0]; print('extracting features from dictionary');
  if l=='p':
   print('using psy.dic'); 
   l_=urllib.request.urlopen('https://raw.githubusercontent.com/facells/learnipy/main/resources/psy.dic').read().decode('utf-8').splitlines();
  elif l=='e':
   print('using emo.dic'); 
   l_=urllib.request.urlopen('https://raw.githubusercontent.com/facells/learnipy/main/resources/emo.dic').read().decode('utf-8').splitlines(); 
  elif l=='d':
   print('using dom.dic'); 
   l_=urllib.request.urlopen('https://raw.githubusercontent.com/facells/learnipy/main/resources/dom.dic').read().decode('utf-8').splitlines(); 
  else:
   print('using custom dictionary'); 
   l_=open(l, encoding="utf8").read().splitlines();
  h=l_.pop(0); h_=h.split(','); t2_=[]; hf=len(h_); #print(dir());
  print(f"dimensions: {h}");
  for t in tqdm(t_):
   t=' '+t+' '; c=t.count(' '); i_=[0 for i_ in range(hf)];
   for i in l_:
    w_=i.split(','); w=w_.pop(0);
    if re.search(w, t):
     for v in w_:
      i_[int(v)]=i_[int(v)]+1;  
   i_=[j/c for j in i_];
   t2_.append(i_);
  t_=PD.DataFrame(t2_,columns=h_); del(t2_);
  if '-d.data' in o:
   print('sync sparse matrix from lexicon:\n',t_)
  else:
   print(f'extracted {hf} features with {l}');




#---feature reduction 
if '-r.svd=' in o: #define dimensions for SVD
 r_=re.findall(r'-r.svd=(.+?) ',o); svdim=int(r_[0]);
 print(f'apply Singular Value Decomposition. Reduction to {svdim} dimensions to obtain dense matrix')
 print('theory: https://en.wikipedia.org/wiki/Singular_value_decomposition');
#else:
# svdim=int(1+(cols/2)); o=f"-r.svd={svdim}"+o;

if '-r.lsa=' in o: #define dimensions for LSA
 r_=re.findall(r'-r.lsa=(.+?) ',o); lsadim=int(r_[0]); 
 print(f'apply Latent Semantic Analysis to reduce to {lsadim} columns dense sync matrix')
 print('theory: https://en.wikipedia.org/wiki/Latent_semantic_analysis');
#else:
# lsadim=50; o=f"-r.lsa={lsadim}"+o;





#---data aggregation
inst=len(x_.index); feat=len(x_.columns); 
if fx==1: #if feature extraction performed concat x_ and t_, else drop t_
 x_=PD.concat([x_, t_], axis=1);
 inst=len(x_.index); feat=len(x_.columns); 
else:
 print('no text feature extraction. text column dropped');
 if feat==0: #x_.empty:
  print('no features. prcess stopped'); inst=len(x_.index); feat=len(x_.columns); 
  print('---END PROCESS---'); sys.exit();

#---check data shape
print(f'dataset shape: {inst} instances, {feat} features');
 

#---class statistics and correlation complexity (only supervised learning)
if '-s.' in o:
 if '-d.viz' in o and not '-t.' in o:
  if task=='s':
   MP.hist(y_, color='black', edgecolor='black', linewidth=0);  
   MP.ylabel('frequency'); MP.title('class distribution');  
   MP.savefig(fname='class-dist'); MP.show(); MP.clf(); #class dist
 if target=='r':
  mu=y_.mean(); mi=y_.min(); ma=y_.max(); sd=y_.std(); me=y_.median(); 
  print(f"min={mi:.1f} max={ma:.1f} avg={mu:.3f} sd={sd:.3f} med={me:.3f} numeric target"); 
  nclass=1;
 if target=='c':
  print(f"class freq"); print(y_.value_counts()); 
  ynp_=y_.to_numpy(); 
  classes, counts=NP.unique(ynp_, return_counts=True); 
  nclass=len(classes); print(f"num classes= {nclass}")
 
 xc_=PD.concat([y_, x_], axis=1); xcc=xc_.corr(); 
 xcc=PD.Series(xcc.iloc[0]); xcc=xcc.iloc[1:]; 
 complexity=1-(xcc.abs().max()); 
 print(f"corr. complexity= {complexity:.3f}"); #compute correlation complexity


#---processed features unsupervised learning: corr, w2v and clustering

if '-u.corm' in o: #correlation matrix
 if'-u.corm=s' in o:
  cort='spearman'
 else:
  cort='pearson'
 if 'y_' in locals():
  x_=PD.concat([x_, y_], axis=1)
 x_=PD.get_dummies(x_); #x_=x_.reset_index(drop=True); 
 print("pearson correlation matrix:\n"+x_.corr(method=cort).to_string()+"\n");
 af= open('log.txt', 'a'); 
 af.write(f"correlation matrix:\n{o}\n"+x_.corr(method=cort).to_string()+"\n\n"); 
 af.close();
 print('theory: https://en.wikipedia.org/wiki/Correlation_coefficient');
 timestamp=DT.datetime.now(); print(f"-u.corm stops other tasks\ntime:{timestamp}"); 
 inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); 
 sys.exit();

if '-u.corr' in o: #correlation list
 #if 'y_' in locals():
 # x_=PD.concat([x_, y_], axis=1)
 x_=PD.get_dummies(x_); #x_=x_.reset_index(drop=True); 
 print("correlation ranks (label values are one-hot encoded):\n");
 #corfound=f'dimensions,rho,pval\n';
 corfound=f"dimensions,rho,pval\n";
 for i in x_:
  xicol=x_[i];
  for j in x_:
   xjcol=x_[j]; 
   if'-u.corr=s' in o:
    corr,pval=ST.spearmanr(xicol, xjcol); #print(f'{corr},{pval}') #run correlation
   else:
    corr,pval=ST.pearsonr(xicol, xjcol);
   #if corr < 0.99 and corr > -0.99 and pval < 0.005 :
   corr=f'{corr:.6f}'; pval=f'{pval:.6f}'; 
   corfound=corfound+f"{xicol.name} and {xjcol.name},{corr},{pval}\n";
 import io; iofile = io.StringIO(corfound);
 cf_ = PD.read_csv(iofile, sep=',');
 cf_=cf_.sort_values(by=['rho'], ascending=False).reset_index(drop=True); 
 print(cf_);
 af= open('log.txt', 'a'); 
 af.write(f"correlation rankings on one-hot values:\n{o}\n"+cf_.to_string()+"\n\n"); 
 af.close();
 print('theory: https://en.wikipedia.org/wiki/Correlation_coefficient');
 timestamp=DT.datetime.now(); print(f"-u.corr stops other tasks\ntime:{timestamp}"); 
 inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();



if '-u.km=' in o: #kmeans clustering
 r_=re.findall(r'-u.km=(.+?) ',o); nk=int(r_[0]);
 clust = SK.cluster.KMeans(n_clusters=nk, random_state=0).fit(x_.values); 
 l_=PD.DataFrame(clust.labels_); 
 l_.columns=['kmeans']; 
 x_=PD.concat([x_,l_], axis=1); 
 g_=x_.groupby('kmeans').mean(); 
 print('applied kmeans clustering. added 1 feature'); 
 print('theory: https://en.wikipedia.org/wiki/K-means_clustering');
 af= open('log.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); 
 print(f"num clusters= {nk}. cluster values:"); print(g_.to_string()+"\n");  
 print("cluster coverage: "); print(+l_.applymap(str).value_counts(normalize=True));
 if '-d.viz' in o:
  projected=pca.fit_transform(x_.values); 
  MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); 
  MP.title('2D data space clusters'); 
  MP.savefig(fname='cluster-space.png'); MP.show(); MP.clf(); #pca space

if '-u.kmpp=' in o: #kmeans++ clustering
 r_=re.findall(r'-u.kmpp=(.+?) ',o); nk=int(r_[0]);
 clust = SK.cluster.MiniBatchKMeans(n_clusters=nk, random_state=0, init='k-means++').fit(x_.values); 
 l_=PD.DataFrame(clust.labels_); 
 l_.columns=['kmeans++']; 
 x_=PD.concat([x_,l_], axis=1); 
 g_=x_.groupby('kmeans++').mean(); 
 print('applied kmeans++ clustering. added 1 feature'); 
 print('theory: https://en.wikipedia.org/wiki/K-means%2B%2B');
 af= open('log.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); 
 print(f"num clusters= {nk}. cluster values:"); print(g_.to_string()+"\n");  
 print("cluster coverage: "); print(+l_.applymap(str).value_counts(normalize=True));
 if '-d.viz' in o:
  projected=pca.fit_transform(x_.values); 
  MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); 
  MP.title('2D data space clusters'); 
  MP.savefig(fname='2d-cluster-space.png'); MP.show(); MP.clf(); #pca space

if '-u.sc=' in o: #spectral clustering
 r_=re.findall(r'-u.sc=(.+?) ',o); nk=int(r_[0]);
 clust = SK.cluster.SpectralClustering(n_clusters=nk, random_state=0, assign_labels='cluster_qr',).fit(x_.values); l_=PD.DataFrame(clust.labels_); l_.columns=['spectral']; x_=PD.concat([x_,l_], axis=1); g_=x_.groupby('spectral').mean(); print('applied spectral clustering. added 1 feature'); 
 print('theory: https://en.wikipedia.org/wiki/Spectral_clustering');
 af= open('log.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); 
 print(f"num clusters= {nk}. cluster values:"); print(g_.to_string()+"\n");  
 print("cluster coverage: "); print(+l_.applymap(str).value_counts(normalize=True));
 if '-d.viz' in o:
  projected=pca.fit_transform(x_.values); 
  MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); 
  MP.title('2D data space clusters'); 
  MP.savefig(fname='2d-cluster-space.png'); MP.show(); MP.clf(); #pca space

if '-u.optics' in o: #optics clustering
 clust = SK.cluster.OPTICS(min_cluster_size=None).fit(x_); 
 l_=PD.DataFrame(clust.labels_); 
 l_.columns=['optics']; x_=PD.concat([x_,l_], axis=1); 
 g_=x_.groupby('optics').mean(); 
 print('applied optics clustering. added 1 feature');
 print('theory: https://en.wikipedia.org/wiki/OPTICS_algorithm');
 af= open('log.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); 
 ynp_=l_.to_numpy(); classes, counts=NP.unique(ynp_, return_counts=True); 
 nk=len(classes); 
 print(f"num clusters= {nk}. cluster values:"); print(g_.to_string()+"\n"); 
 print("cluster coverage: "); print(+l_.applymap(str).value_counts(normalize=True));
 if '-d.viz' in o:
  projected=pca.fit_transform(x_.values); 
  MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); 
  MP.title('2D data space clusters'); 
  MP.savefig(fname='2d-cluster-space.png'); MP.show(); MP.clf(); #pca space

if '-u.msh' in o: #meanshift clustering
 clust = SK.cluster.MeanShift().fit(x_); l_=PD.DataFrame(clust.labels_); 
 l_.columns=['mshift']; x_=PD.concat([x_,l_], axis=1); 
 g_=x_.groupby('mshift').mean(); 
 print('applied mshift clustering. added 1 feature');
 print('theory: https://en.wikipedia.org/wiki/Mean_shift');
 af= open('log.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); 
 ynp_=l_.to_numpy(); classes, counts=NP.unique(ynp_, return_counts=True); 
 nk=len(classes); 
 print(f"num clusters= {nk}. cluster values:"); print(g_.to_string()+"\n");  
 print("cluster coverage: "); print(+l_.applymap(str).value_counts(normalize=True));
 if '-d.viz' in o:
  projected=pca.fit_transform(x_.values); 
  MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); 
  MP.title('2D data space clusters'); 
  MP.savefig(fname='2d-cluster-space.png'); MP.show(); MP.clf(); #pca space


if '-u.som' in o: #self organising map clustering (contributor: Fabio Celli)
 os.system('pip install minisom'); from minisom import MiniSom; #library
 xn_=x_.to_numpy(); xn_=NP.array(xn_).astype(float); #change dataset x_ format to numpy to  use minisom
 nodes=int(4*math.sqrt(len(xn_))); #compute nodes of the matrix based on instances
 clust = MiniSom(nodes, nodes, feat, sigma=0.3, learning_rate=0.5, random_seed=2) 
 clust.train(xn_, 100) #train SOM with 100 iterations
 l_=[]; col_=[]; #new column with som results
 for i in xn_:
  w=clust.winner(i); 
  l=(w[0]+w[1])/(nodes*2);l_.append(l); #compute som results, put in l_
  col=int(w[0]+w[1]);col_.append(col); #compute int for colors, put in col_
 l_=PD.DataFrame(l_); l_.columns=['som']; x_=PD.concat([x_,l_], axis=1); #print(l_)
 g_=x_.groupby('som').mean(); 
 print(f'apply self organizing maps with {nodes} x {nodes} matrix. added 1 feature');
 print('theory: https://en.wikipedia.org/wiki/Self-organizing_map');
 af= open('log.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); 
 ynp_=l_.to_numpy(); classes, counts=NP.unique(ynp_, return_counts=True); 
 nk=len(classes);  print(f"num clusters= {nk}. cluster values:"); 
 print(g_.to_string()+"\n");  print("cluster coverage: "); 
 print(+l_.applymap(str).value_counts(normalize=True));
 if '-d.viz' in o:
  projected=pca.fit_transform(x_.values); 
  MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(col_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); 
  MP.colorbar(); MP.title('2D PCA data space som clusters'); 
  MP.savefig(fname='pca-cluster-space.png'); MP.show(); MP.clf(); #pca space

if '-u.ap' in o: #affinity propagation clustering
 clust = SK.cluster.AffinityPropagation(damping=0.5).fit(x_); 
 l_=PD.DataFrame(clust.labels_); l_.columns=['affinity']; 
 x_=PD.concat([x_,l_], axis=1); g_=x_.groupby('affinity').mean(); 
 print('applied affinity propagation clustering. added 1 feature');
 print('theory: https://en.wikipedia.org/wiki/Affinity_propagation');
 af= open('log.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); 
 ynp_=l_.to_numpy(); classes, counts=NP.unique(ynp_, return_counts=True); 
 nk=len(classes); 
 print(f"num clusters= {nk}. cluster values:"); print(g_.to_string()+"\n");  
 print("cluster coverage: "); print(+l_.applymap(str).value_counts(normalize=True));
 if '-d.viz' in o:
  projected=pca.fit_transform(x_.values); 
  MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); 
  MP.title('2D data space clusters'); 
  MP.savefig(fname='2d-cluster-space.png'); MP.show(); MP.clf(); #pca space


#---exporting
if '-d.export' in o:
 r_=re.findall(r'-d.export=(.+?) ',o); xfn=r_[0];
 y_=y_.rename('class');
 n_=PD.concat([x_,y_], axis=1); print('exporting processed dataset'); 
 af= open(f"{xfn}", 'w'); af.write(n_.to_csv()); af.close();  
 print(f"data saved as {xfn}");
 #inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();

#---preprocessing features
if not '-x.ts=' in o or not 't_' in locals():
 maxval=2;
#column normalization range 0-1 (required for sgd and nb)
 if '-p.fn' in o or '-s.sgd' in o or '-s.nb' in o or '-s.nn' in o: 
  x_scaled=SK.preprocessing.MinMaxScaler().fit_transform(x_.values); 
  x_=PD.DataFrame(x_scaled, columns=x_.columns); 
  print('apply feature normalization as needed'); #normalization by column
if '-x.ts=' in o:
 #x_=x_.div(x_.sum(axis=1), axis=0); print('apply instance normalization'); #normalize by row
 #x_scaled=SK.preprocessing.MinMaxScaler().fit_transform(x_.values); 
 #x_=PD.DataFrame(x_scaled, columns=x_.columns); 
 #print('apply feature normalization as needed'); #normalization by column
 maxval=1000;

print(f"max value for neural network embedding= {maxval}");

#*TO REMOVE*
#if '-r.anova' in o and task=='s':
# rowsize=int(len(x_.index)/5); colsize=int(len(x_.columns)/5);  size=min([rowsize,colsize]);
# if target=='c':
#  fs_=SK.feature_selection.SelectKBest(SK.feature_selection.f_classif, k=size);  xfs_=fs_.fit_transform(x_, y_); names = x_.columns[fs_.get_support()]; 
# if target=='r':
#  fs_=SK.feature_selection.SelectKBest(SK.feature_selection.f_regression, k=size);  xfs_=fs_.fit_transform(x_, y_); names = x_.columns[fs_.get_support()]; 
# x_=PD.DataFrame(xfs_, columns=names); #supervised feature rank reduction with chi2. classification:f_classif, mutual_info_classif; regression: f_regression, mutual_info_regression
# feat=len(x_.columns); print(f'applied anova feature selection, reduced to {feat} features');
#
#if '-r.mi' in o and task=='s':
# rowsize=int(len(x_.index)/5); colsize=int(len(x_.columns)/5);  size=min([rowsize,colsize]);
# if target=='c':
#  fs_=SK.feature_selection.SelectKBest(SK.feature_selection.mutual_info_classif, k=size);  xfs_=fs_.fit_transform(x_, y_); names = x_.columns[fs_.get_support()]; 
# if target=='r':
#  fs_=SK.feature_selection.SelectKBest(SK.feature_selection.mutual_info_regression, k=size);  xfs_=fs_.fit_transform(x_, y_); names = x_.columns[fs_.get_support()]; 
# x_=PD.DataFrame(xfs_, columns=names); 
# size=len(x_.columns); print(f'applied mutual info feature selection, reduced to {size} features');


#---feature summary
x_=PD.DataFrame(x_.values.astype(NP.float64), columns=x_.columns); 
print('turn all features to float numbers.'); #print(x_);
#print(list(x_)); #print(x_.dtypes);
sparseness=0; sparsemsg=''; 
sparse = sum((x_ == 0).astype(int).sum())/x_.size; #initialize sparseness check
if sparse>=0.9:
 sparseness=1; sparsemsg='WARNING: high data sparseness, better to apply feature reduction. ';
print(f"sparseness= {sparse:.3f}");

feature_names = x_.columns;

x_.columns = range(x_.shape[1]); #make columns unique 


if '-d.fdist' in o:
 #print(x_.dtypes); print(y_.dtypes);
 for col in x_:
  mu=x_[col].mean(); 
  mi=x_[col].min(); 
  ma=x_[col].max(); 
  me=x_[col].median(); 
  sd=x_[col].std(); 
  print(f"min={mi:.1f} max={ma:.1f} avg={mu:.3f} sd={sd:.3f} med={me:.3f} for {col}"); 




#---predictions with saved model
if '.h4' in f: #apply machine learning saved model
 #print(y_); #print(len(y_)); #print(loadmodel.k_ar); 
 if '-t.' in o:
  if '-t.a' in o:
   y2_pred=loadmodel.forecast(steps=100)[0];
  else:
   y2_pred=loadmodel.forecast(steps=100); 
  #print(y2_pred); print(o)
  y2_pred=PD.DataFrame(y2_pred); 
  n_=PD.concat([y_,y2_pred], axis=0).reset_index(); 
  print('applied saved model on supplied dataset. results:\n'); print(n_);
  af= open(f"{testname}-predictions.csv", 'w'); 
  af.write(n_.to_csv()); af.close();  
  print(f"data with predictions saved as {testname}-predictions.csv");
  inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();  
 else:
  y2_pred=loadmodel.predict(x_); y2_pred=PD.DataFrame(y2_pred); #print(type())
  n_=PD.concat([x2_,y2_pred], axis=1); 
  print('applied saved model on supplied dataset. results:\n'); print(n_);
  af= open(f"{testname}-predictions.csv", 'w'); af.write(n_.to_csv()); af.close();  
  print(f"data with predictions saved as {testname}-predictions.csv");
  inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();

if '.h5' in f: #apply deep learning saved model
 if '-t.' in o: #in case of timeseries
  d_=PD.to_datetime(d_); y_=PD.DataFrame(y_); 
  y_=PD.concat([y_,d_], axis=1); y_['date']=d_; 
  y_.sort_values(by="date"); # sort x_ in date order
  y_=y_.drop(columns=['date']); y_['date']=y_.index;  #substitute date with integer index
  scaler = SK.preprocessing.MinMaxScaler(feature_range=(0, 1)); 
  y_=y_.drop(columns=['date']); y_ = scaler.fit_transform(y_)
  y_ = NP.reshape(y_, (y_.shape[0], y_.shape[1], 1));
  y2_pred=loadmodel.predict(y_); 
  y2_pred = scaler.inverse_transform(y2_pred);
 else:
  if datatype=='zip':
   x_=x_.to_numpy(); print(x_.shape);
   x_=x_.reshape(-1,dshape[0],dshape[1],3);
  y2_pred=loadmodel.predict(x_);
 if target=='c':
  y2_pred=PD.DataFrame(y2_pred); y2_pred=y2_pred.idxmax(axis=1); #print(y2_pred);
 if target=='r':
  y2_pred=PD.DataFrame(y2_pred);
 n_=PD.concat([x2_,y2_pred], axis=1); print('applied saved model on supplied test set'); 
 af= open(f"{testname}-predictions.csv", 'w'); af.write(n_.to_csv()); af.close(); 
 print(f"data with predictions saved as {testname}-predictions.csv");
 x_=PD.DataFrame(x_); 
 inst=len(x_.index); feat=len(x_.columns); 
 print('---END PROCESS---'); sys.exit();

#---stop unsupervised learning
if task=='u':
 print('no supervised or timeseries algorithms to run'); 
 inst=len(x_.index); feat=len(x_.columns); 
 print('---END PROCESS---'); sys.exit();

#---train and test split
#if '-e.cv=' in o: # cross validation *TO REMOVE*
# r_=re.findall(r'-e.cv=(.+?) ',o); folds=int(r_[0]); cv = SK.model_selection.RepeatedKFold(n_splits=folds, random_state=1); # prepare the cross-validation eval

if '-e.tts' in o:
 r_=re.findall(r'-e.tts=(.+?) ',o); split=float(r_[0]); 
 if 'x2_' in locals():
  split=tts;
else:
  split=0.3; print('using 70% train 30% test percentage split by default');
if 'split' in locals(): #if split percentage is defined, then split train and test sets
 if '-t.' in o: #in case of timeseries
  d_=PD.to_datetime(d_); y_=PD.DataFrame(y_);
  #x_=PD.concat([d_, x_], axis=1); y_=PD.concat([d_, y_], axis=1); #put timestamps to train and test
  x_['date']=d_; x_.sort_values(by="date"); # sort x_ in date order
  y_['date']=d_; y_.sort_values(by="date"); # sort x_ in date order
  tscol='date';
  x_train, x_test, y_train, y_test=SK.model_selection.train_test_split(x_, y_, test_size=split, shuffle=False) # prepare train test eval 
 else:
  x_train, x_test, y_train, y_test=SK.model_selection.train_test_split(x_, y_, test_size=split, shuffle=False, random_state=1) # prepare train test eval 
 xtrain_inst=len(x_train.index); feat=len(x_train.columns); 
 print(f'training set shape: {xtrain_inst} instances, {feat} features');
 xtest_inst=len(x_test.index); feat=len(x_test.columns); 
 print(f'test set shape: {xtest_inst} instances, {feat} features');
 x_test2=x_test; #create a copy of x_test for evaluation in case of shape change
 print(f"training set stats"); print(y_train.describe());
 print(f"test set stats"); print(y_test.describe());


if '-d.data' in o:
 if '-t.' in o: #in case of timeseries
  print('data ready for learning:'); print(y_);  
 else:
  print('data ready for learning:'); print(x_);

#---outlier detection
if '-o.' in o:
 print(f"dataset before outlier detection: training set {x_train.shape}")
 if target=='c':
  model=SK.dummy.DummyClassifier(strategy='prior'); model.fit(x_train, y_train); y_pred=model.predict(x_test);
  acc=SK.metrics.f1_score(y_test, y_pred); 
  print(f"baseline before outlier detection: F1= {acc:.3f}"); 
 if target=='r':
  model=SK.dummy.DummyRegressor(strategy='mean'); model.fit(x_train, y_train); y_pred=model.predict(x_test);
  r2=SK.metrics.r2_score(y_test, y_pred);
  print(f"baseline before outlier detection: R2= {r2:.3f}");

 #apply algorithms
 if '-o.if' in o:
  print('apply isolation forest\ntheory: https://en.wikipedia.org/wiki/Isolation_forest')
  iso = SK.ensemble.IsolationForest(contamination=0.1)
  yhat = iso.fit_predict(x_train)
  mask = yhat != -1;
  x_train, y_train = x_train[mask], y_train[mask]; # select all rows that are not outliers
  print(f"dataset after outlier detection: training set {x_train.shape}")
 
 if '-o.mcd' in o:
  print('apply minimum covariance determinant\ntheory: https://en.wikipedia.org/wiki/Covariance_matrix#Covariance_matrix_as_a_parameter_of_a_distribution')
  ee = SK.covariance.EllipticEnvelope(contamination=0.01)
  yhat = ee.fit_predict(x_train) 
  mask = yhat != -1; 
  x_train, y_train = x_train[mask], y_train[mask]; # select all rows that are not outliers
  print(f"dataset after outlier detection: training set {x_train.shape}")

 if '-o.lof' in o:
  print('apply local outlier factor\ntheory: https://en.wikipedia.org/wiki/Local_outlier_factor')
  lof = SK.neighbors.LocalOutlierFactor()
  yhat = lof.fit_predict(x_train)
  mask = yhat != -1; 
  x_train, y_train = x_train[mask], y_train[mask]; # select all rows that are not outliers
  print(f"dataset after outlier detection: training set {x_train.shape}")

 #evaluate
 if target=='c':
  model=SK.dummy.DummyClassifier(strategy='prior'); 
  model.fit(x_train, y_train); y_pred=model.predict(x_test);
  acc=SK.metrics.f1_score(y_test, y_pred); 
  print(f"baseline after outlier detection: F1= {acc:.3f}"); 
 if target=='r':
  model=SK.linear_model.LinearRegression(); 
  model.fit(x_train, y_train); y_pred=model.predict(x_test);
  r2=SK.metrics.r2_score(y_test, y_pred);
  print(f"baseline after outlier detection: R2= {r2:.3f}");

from sklearn.inspection import permutation_importance

#---supervised learning
if '-s.base' in o and target=='c':
 model=SK.dummy.DummyClassifier(strategy='prior', random_state=3,); 
 print('evaluate with baseline dummy classifier'); 
 model.fit(x_train, y_train); y_pred=model.predict(x_test); 
if '-s.base' in o and target=='r':
 model=SK.dummy.DummyRegressor(strategy='mean', random_state=3,); 
 print('evaluate with mean baseline');
 model.fit(x_train, y_train); y_pred=model.predict(x_test);  #'mean', 'median',

if '-s.nb' in o and target=='c':
 model=SK.naive_bayes.ComplementNB();model.fit(x_train, y_train); 
 y_pred=model.predict(x_test); 
 imprank = permutation_importance(model, x_train, y_train, scoring='accuracy').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
 print('apply complement naive bayes classification (on normalized space)\ntheory: https://en.wikipedia.org/wiki/Naive_Bayes_classifier');
if '-s.nb' in o and target=='r':
 model=SK.linear_model.BayesianRidge();model.fit(x_train, y_train); 
 y_pred=model.predict(x_test); y_pred=y_pred.flatten(); 
 imprank = permutation_importance(model, x_train, y_train, scoring='neg_mean_squared_error').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
 print('apply bayesian ridge regression (on normalized space)\ntheory: https://en.wikipedia.org/wiki/Bayesian_linear_regression');

if '-s.lcm' in o and target=='c':
 model=SK.discriminant_analysis.LinearDiscriminantAnalysis();
 model.fit(x_train, y_train); y_pred=model.predict(x_test); 
 imprank = permutation_importance(model, x_train, y_train, scoring='accuracy').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
 print('apply LinearDiscriminantAnalysis classification \ntheory: https://en.wikipedia.org/wiki/Linear_discriminant_analysis');
if '-s.lcm' in o and target=='r':
 model=SK.cross_decomposition.PLSRegression(max_iter=500);
 model.fit(x_train, y_train); 
 imprank = permutation_importance(model, x_train, y_train, scoring='neg_mean_squared_error').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
 y_pred=model.predict(x_test); y_pred=y_pred.flatten(); 
 print('apply PartialLeastSquare regression \ntheory: https://en.wikipedia.org/wiki/Partial_least_squares_regression');

if '-s.lr' in o and target=='c':
 model=SK.linear_model.LogisticRegression(max_iter=5000);
 model.fit(x_train, y_train); y_pred=model.predict(x_test); 
 imprank = PD.Series(model.coef_[0], index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 print('apply logistic regression classification \ntheory:https://en.wikipedia.org/wiki/Logistic_regression');
if '-s.lr' in o and target=='r':
 model=SK.linear_model.LinearRegression();
 model.fit(x_train, y_train); y_pred=model.predict(x_test); 
 imprank = PD.Series(model.coef_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 print('apply linear regression \ntheory: https://en.wikipedia.org/wiki/Linear_regression');

if '-s.sgd' in o and target=='c':
 model=SK.linear_model.SGDClassifier(shuffle=False);
 model.fit(x_train, y_train); y_pred=model.predict(x_test); 
 imprank = PD.Series(model.coef_[0], index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 print('apply stochastic gradient descent classification (on normalized space) \ntheory: https://en.wikipedia.org/wiki/Stochastic_gradient_descent');
if '-s.sgd' in o and target=='r':
 model=SK.linear_model.SGDRegressor(shuffle=False);
 model.fit(x_train, y_train); y_pred=model.predict(x_test); 
 imprank = PD.Series(model.coef_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 print('apply sochastic gradient descent regression (on normalized space) \ntheory: https://en.wikipedia.org/wiki/Stochastic_gradient_descent');

if '-s.knn' in o and target=='c':
 model=SK.neighbors.KNeighborsClassifier();model.fit(x_train, y_train); 
 y_pred=model.predict(x_test); 
 imprank = permutation_importance(model, x_train, y_train, scoring='accuracy').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
 print('apply k nearest neighbors classification \ntheory: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm');
if '-s.knn' in o and target=='r':
 model=SK.neighbors.KNeighborsRegressor();model.fit(x_train, y_train); 
 y_pred=model.predict(x_test); 
 imprank = permutation_importance(model, x_train, y_train, scoring='neg_mean_squared_error').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
 print('apply k nearest neighbors regression \ntheory: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm');

if '-s.mlp' in o and target=='c':
 model=SK.neural_network.MLPClassifier(random_state=1);
 model.fit(x_train, y_train); 
 y_pred=model.predict(x_test); 
 imprank = permutation_importance(model, x_train, y_train, scoring='accuracy').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
 print('apply multi layer perceptron classification \ntheory: https://en.wikipedia.org/wiki/Multilayer_perceptron');
if '-s.mlp' in o and target=='r':
 model=SK.neural_network.MLPRegressor(random_state=1);
 model.fit(x_train, y_train); 
 y_pred=model.predict(x_test); 
 imprank = permutation_importance(model, x_train, y_train, scoring='neg_mean_squared_error').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
 print('apply multi layer perceprtron regression \ntheory: https://en.wikipedia.org/wiki/Multilayer_perceptron');

if '-s.svm' in o and target=='c':
 print('apply support vector machines\ntheory: https://en.wikipedia.org/wiki/Support-vector_machine');
 kern='r'; mi=5;
 if '-s.svm=' in o:
  r_=re.findall(r'-s.svm=(.+?) ',o);
  kern=r_[0][0]; mi=int(r_[0][1]);
 if kern=='p':
  model=SK.svm.NuSVC(kernel='poly', degree=mi);
  model.fit(x_train, y_train); y_pred=model.predict(x_test);
  imprank = permutation_importance(model, x_train, y_train, scoring='accuracy').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
  print('using polynomial kernel\ntheory: https://en.wikipedia.org/wiki/Polynomial_kernel')
 if kern=='r':
  model=SK.svm.NuSVC(kernel='rbf', nu=mi/10);
  model.fit(x_train, y_train);
  y_pred=model.predict(x_test); 
  imprank = permutation_importance(model, x_train, y_train, scoring='accuracy').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
  print('using rbf kernel\ntheory: https://en.wikipedia.org/wiki/Radial_basis_function_kernel')
if '-s.svm' in o and target=='r':
 print('apply support vector machines\ntheory: https://en.wikipedia.org/wiki/Support-vector_machine');
 kern='r'; mi=5;
 if '-s.svm=' in o:
  r_=re.findall(r'-s.svm=(.+?) ',o); 
  kern=r_[0][0]; mi=int(r_[0][1]);
 if kern=='p':
  model=SK.svm.NuSVR(kernel='poly', degree=mi);
  model.fit(x_train, y_train); 
  y_pred=model.predict(x_test);
  imprank = permutation_importance(model, x_train, y_train, scoring='neg_mean_squared_error').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
  print(f"using polynomial kernel={mi}\ntheory: https://en.wikipedia.org/wiki/Polynomial_kernel")
 if kern=='r':
  model=SK.svm.NuSVR(kernel='rbf', nu=mi/10);
  model.fit(x_train, y_train);
  y_pred=model.predict(x_test); 
  imprank = permutation_importance(model, x_train, y_train, scoring='neg_mean_squared_error').importances_mean; imprank = PD.Series(imprank, index=feature_names).sort_values(ascending=False); print(f"permutation feature importance:\n{imprank}")
  print(f"using rbf kernel={mi/10}\ntheory: https://en.wikipedia.org/wiki/Radial_basis_function_kernel")

if '-s.rf' in o and target=='c':
 model=SK.ensemble.RandomForestClassifier(random_state=1);
 model.fit(x_train, y_train); 
 imprank = PD.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 y_pred=model.predict(x_test); 
 print('apply random forest classification\ntheory: https://en.wikipedia.org/wiki/Random_forest');
if '-s.rf' in o and target=='r':
 model=SK.ensemble.RandomForestRegressor(random_state=1);
 model.fit(x_train, y_train);
 imprank = PD.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 y_pred=model.predict(x_test); 
 print('apply random forest regression\ntheory: https://en.wikipedia.org/wiki/Random_forest');

if '-s.ada' in o and target=='r':
 model=SK.ensemble.AdaBoostRegressor(random_state=1);
 model.fit(x_train, y_train); 
 imprank = PD.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 y_pred=model.predict(x_test); 
 print('apply adaboost regression\ntheory: https://en.wikipedia.org/wiki/AdaBoost')
if '-s.ada' in o and target=='c':
 model=SK.ensemble.AdaBoostClassifier(random_state=1);
 model.fit(x_train, y_train);
 imprank = PD.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 y_pred=model.predict(x_test); 
 print('apply adaboost classification\ntheory: https://en.wikipedia.org/wiki/AdaBoost');

if '-s.dt' in o:
 prune=0
 if '-s.dt=' in o: 
  r_=re.findall(r'-s.dt=(.) ',o); 
  if int(r_[0])>0:
   prune=(int(r_[0])/100);
 if target=='c':
  model=SK.tree.DecisionTreeClassifier(ccp_alpha=prune);
  model.fit(x_train, y_train); 
  imprank = PD.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
  y_pred=model.predict(x_test); 
  print('apply decision trees classification\ntheory https://en.wikipedia.org/wiki/Decision_tree_learning'); 
 if target=='r':
  model=SK.tree.DecisionTreeRegressor(ccp_alpha=prune);
  model.fit(x_train, y_train); 
  imprank = PD.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
  y_pred=model.predict(x_test); 
  print('apply decision trees regression\ntheory https://en.wikipedia.org/wiki/Decision_tree_learning'); 

if '-s.xgb' in o and target=='c':
 import xgboost; model=xgboost.XGBClassifier();
 model.fit(x_train, y_train); 
 imprank = PD.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 y_pred=model.predict(x_test); 
 print('apply gradient boosting classification\ntheory: https://en.wikipedia.org/wiki/Gradient_boosting');
if '-s.xgb' in o and target=='r':
 import xgboost; model=xgboost.XGBRegressor();
 model.fit(x_train, y_train); 
 imprank = PD.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False); print(f"feature importance:\n{imprank}")
 y_pred=model.predict(x_test); 
 print('apply gradient boosting regression\ntheory: https://en.wikipedia.org/wiki/Gradient_boosting');


if 'model' in locals() and '-s.' in o:
 if target=='c': #compute classes for machine learning
  classes=model.classes_;
 if '-d.save' in o and not '-s.nn' in o: #save machine learning models
  opt=re.sub(r'-d.save|-e\..+|-d.viz| ','', o); 
  if target=='c':
   opt='-d.t=c'+opt;
  else:
   opt='-d.t=r'+opt;
  joblib.dump(model, f"{filename}{opt}.h4");
  print(f"model saved as {filename}{opt}.h4");
  #af= open(f"{filename}-format4model.csv", 'w'); af.write(x_test.to_csv()); af.close();  



if '-s.nn' in o:
 x_train=x_train.to_numpy(); x_test=x_test.to_numpy(); 
 y_train=y_train.to_numpy(); y_test=y_test.to_numpy(); #turn dataframe to numpy
 print('apply deep learning neural networks\ntheory: https://en.wikipedia.org/wiki/Deep_learning');
 #l2=2; nu=5; 
 nl=int((complexity/2)*10); nu=int(math.sqrt(feat * nclass)); #automatic selection of num. layers and nodes

 r_=re.findall(r'-s.nn=(.+?) ',o); 
 if len(r_[0]) > 1:
  nt=r_[0][0]; nu=int(r_[0][1]); nl=int(r_[0][2]);  
  print('using neural network options')
 else:
  print('no neural network options given. detecting best settings:');

 #nu=2**nu; l2=0.1**l2; #compute optional values
 
 #print(x_train); print(y_train); print(x_train.shape);

 if sparseness==1: #if data is sparse
  opt='adadelta';
 else: #in normal conditions
  opt='adam';
 if target=='c': # if classification,
  metric='accuracy'; outactiv='sigmoid'; los='sparse_categorical_crossentropy'; #compute classes and options for deep learning
 else: # if regression
  outactiv='linear'; metric='mae'; los='mse';
 if 'nn=l' in o or 'nn=b' in o or 'nn=g' in o or 'nn=r' in o:
  activ='tanh'
 else:
  activ='selu'

 print(f"layer cofig: num={nl}, actv={activ}, nodes={nu}"); 
 print(f"network config: optimizer={opt}, out_actv={outactiv}");
 print(f"validator config: loss={los}, metric={metric},");
 #optimizers: adam(robust) sgd(fast) rmsprop(variable) adadelta(for sparse data)
 #activations: linear(lin,-inf/+inf) gelu(nonlin,-0.17/+inf) selu(nonlin,-a/+inf) sigmoid(nonlin,0/1) tanh(nonlin,-1/+1) softplus(nonlin,0/+inf) softsign(nonlin,-1/+1). linear=regression, relu|selu|gelu=general purpose, sigmoid|tanh=binary classification, softplus|softsign=multiclass classifiction
 #losses: hinge kld cosine_similarity mse msle huber binary_crossentropy sparse_categorical_crossentropy
 #metrics: mape mae accuracy top_k_categorical_accuracy categorical_accuracy

 if '-s.nn=o' in o: #feedforward
  model=TF.keras.Sequential();
  model.add(TF.keras.layers.Dense(nclass, activation=outactiv)); #output nodes are=nclass
 if '-s.nn=f' in o: #feedforward
  model=TF.keras.Sequential();
  model.add(TF.keras.layers.Dense(feat, activation=activ)); #initial nodes are=num features
  model.add(TF.keras.layers.Dense(nu*2, activation=activ));
  [model.add(TF.keras.layers.Dense(nu, activation=activ)) for n in range(0,nl)]
  model.add(TF.keras.layers.Dense(int(nu/2), activation=activ));  
  model.add(TF.keras.layers.Dense(nclass, activation=outactiv)); #output nodes are=nclass
 if '-s.nn=i' in o: #imbalancenet
  model=TF.keras.Sequential();
  model.add(TF.keras.layers.Dense(feat, activation=activ)); #initial nodes are=num features
  model.add(TF.keras.layers.Dense(nu*2, activation=activ));
  model.add(TF.keras.layers.Dropout(0.3));
  [model.add(TF.keras.layers.Dense(nu*2, activation=activ)) for n in range(0,nl)]
  model.add(TF.keras.layers.Dropout(0.3));
  model.add(TF.keras.layers.Dense(nclass, activation=outactiv)); #output nodes are=nclass
 if '-s.nn=l' in o: #lstm
  model=TF.keras.Sequential();
  model.add(TF.keras.layers.Embedding(maxval,32,input_length=feat));
  [model.add(TF.keras.layers.LSTM(nu, activation=activ, return_sequences=True)) for n in range(0,nl)]
  model.add(TF.keras.layers.LSTM(nu, activation=activ)) 
  model.add(TF.keras.layers.Dense(nclass, activation=outactiv));
 if '-s.nn=b' in o: #bilstm
  model = TF.keras.Sequential()
  model.add(TF.keras.layers.Embedding(maxval,nu,input_length=feat))
  [model.add(TF.keras.layers.Bidirectional(TF.keras.layers.LSTM(int(nu/2), activation=activ, return_sequences=True))) for n in range(0,nl)]
  model.add(TF.keras.layers.Bidirectional(TF.keras.layers.LSTM(int(nu/2), activation=activ)))
  model.add(TF.keras.layers.Dense(nclass, activation=outactiv))
 if '-s.nn=g' in o: #gru
  model = TF.keras.Sequential();
  model.add(TF.keras.layers.Embedding(maxval,nu,input_length=feat))
  [model.add(TF.keras.layers.GRU(nu*2, activation=activ, return_sequences=True)) for n in range(0,nl)]
  model.add(TF.keras.layers.GRU(nu, activation=activ))
  model.add(TF.keras.layers.Dense(nclass, activation=outactiv))
 if '-s.nn=r' in o: #rnn
  model = TF.keras.Sequential();
  model.add(TF.keras.layers.Embedding(maxval,nu,input_length=feat))
  [model.add(TF.keras.layers.SimpleRNN(nu, activation=activ, return_sequences=True)) for n in range(0,nl)]
  model.add(TF.keras.layers.SimpleRNN(nu, activation=activ))
  model.add(TF.keras.layers.Dense(nclass, activation=outactiv))
 if '-s.nn=c' in o: #cnn
  if len(dshape)==1: #for tables
   model = TF.keras.Sequential();
   model.add(TF.keras.layers.Embedding(maxval,nu,input_length=feat))
   [model.add(TF.keras.layers.Conv1D(int(nu/2), 7, activation=activ,padding='same' )) for n in range(0,nl)]
   #model.add(TF.keras.layers.Conv1D(32, 7, activation=activ,padding='same')); 
   model.add(TF.keras.layers.GlobalMaxPooling1D());
   model.add(TF.keras.layers.Dropout(0.3));
   model.add(TF.keras.layers.Dense(nclass, activation=outactiv));
  if len(dshape)==2: #for images
   x_train=x_train.reshape((xtrain_inst,dshape[0],dshape[1],3));
   x_test=x_test.reshape((xtest_inst,dshape[0],dshape[1],3));
   model = TF.keras.Sequential();
   [model.add(TF.keras.layers.Conv2D(nu, kernel_size=(3, 3), activation=activ, input_shape=(dshape[0],dshape[1],3))) for n in range(0,nl)]
   model.add(TF.keras.layers.MaxPooling2D(pool_size=(2, 2)));
   model.add(TF.keras.layers.Conv2D(nu*2, kernel_size=(3, 3), activation=activ));
   model.add(TF.keras.layers.MaxPooling2D(pool_size=(2, 2)));
   model.add(TF.keras.layers.Flatten()); model.add(TF.keras.layers.Dropout(0.5));
   model.add(TF.keras.layers.Dense(nclass, activation=outactiv));

 
 model.compile(optimizer=opt,  loss=los,  metrics=[metric]);
 
 print('creating models on training set. max 100 epochs, stop after 5 epochs with no improvement');
 earlystop = TF.keras.callbacks.EarlyStopping(patience=5, monitor=metric); 
 model.fit(x_train, y_train, epochs=100, callbacks=[earlystop], verbose=1);# 

 print('evaluating model on test set'); model.evaluate(x_test,  y_test, verbose=2); 
 if target=='c':
  y_pred=model.predict(x_test); y_pred = y_pred.argmax(axis=-1); y_test=PD.Series(y_test);   #make class predictions from functional model and count classes from test for conf matrix
 #print(y_pred);
 if target=='r':
  y_pred=model.predict(x_test); y_pred=y_pred.flatten(); #print(y_pred); 
  y_test=PD.Series(y_test); #ymin=y_train.min(); ymax=y_train.max(); print(f"ymin={ymin} ymax={ymax}"); #get weights from model
  #y_test=PD.Series(y_test); diff=y_pred.flatten()-y_test; modeldiff=(diff / y_test)*y_test.mean(); y_pred=NP.abs(modeldiff); #make numeric predictions from functional model
  y_pred=PD.DataFrame(y_pred).to_numpy().flatten(); 

 if '-d.save' in o and '-s.nn' in o: #save deep learning models
  opt=re.sub(r'-d.save|-e\..+|-d.viz| ','', o); 
  if target=='c':
   opt='-d.t=c'+opt;
  if target=='r':
   opt=f"-d.t=r"+opt;
  model.save(f"{filename}{opt}.h5"); print(f"model saved as {filename}{opt}.h5");


#---time series forecasting
if '-t.' in o:
 dy_=y_test[tscol]; dy_=PD.Series(dy_);# take dates of test set
 x_train= x_train.set_index(tscol); y_train= y_train.set_index(tscol);  #set date as the instance index
 x_test= x_test.set_index(tscol); y_test= y_test.set_index(tscol);  #set date as the instance index
 #print(dir(SM.tsa));

#if '-t.arma' in o:
# model=SM.tsa.ARMA(y_train, order=(3, 2, 1)).fit();# fit model
# y_pred=model.predict(len(y_train), len(y_train)+len(y_test)-1)# make prediction
# y_pred=y_pred.to_numpy();
# print(f"using AutoReg Moving Average for time series\ntheory https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model")
if '-t.arima' in o: 
 model=SM.tsa.ARIMA(y_train, order=(3, 2, 1)).fit();# fit model
 y_pred=model.predict(len(y_train), len(y_train)+len(y_test)-1, typ='levels')# make prediction
 y_pred=y_pred.to_numpy();
 print(f"using AutoReg Integrated Moving Average for time series\ntheory https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average")
if '-t.sarima' in o: 
 model = SM.tsa.SARIMAX(y_train, order=(3, 2, 1)).fit(disp=False)
 y_pred=model.predict(len(y_train), len(y_train)+len(y_test)-1)# make prediction
 y_pred=y_pred.to_numpy();
 print(f"using Seasonal AutoReg Integrated Moving Average for time series\ntheory https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average")
if '-t.hwes' in o: 
 model = SM.tsa.Holt(y_train).fit()
 y_pred=model.predict(len(y_train), len(y_train)+len(y_test)-1)# make prediction
 y_pred=y_pred.to_numpy();
 print(f"using HoltWinters Exponential Smoothing for time series\ntheory https://en.wikipedia.org/wiki/Exponential_smoothing")


if 'model' in locals() and '-t.' in o:
 if target=='c': 
  classes=model.classes_;
 if '-d.save' in o and not '-t.nn' in o: #save machine learning models
  opt=re.sub(r'-d.save|-e\..+|-d.viz| ','', o); 
  if target=='c':
   opt='-d.t=c'+opt;
  else:
   opt='-d.t=r'+opt;
  joblib.dump(model, f"{filename}{opt}.h4");print(f"model saved as {filename}{opt}.h4");
  #af= open(f"{filename}-format4model.csv", 'w'); af.write(x_test.to_csv()); af.close();  

if not 'y_pred' in locals():
 print('no test data to evaluate'); inst=len(x_.index); feat=len(x_.columns); print('---END PROCESS---'); sys.exit();

#---model details
if '-d.md' in o and not '-s.base' in o:
 print('model details:');
 #print(dir(model));
 if 'weights' in dir(model) and not '.knn' in o: #nn
  mw=f" per.layer.node.weights:\n{model.weights}"; 
 if 'coef_' in dir(model) and not 'support_vectors_' in dir(model) and not '-s.xgb' in o: #lr, lcm, sgd, nb 
  mw=f" features * per.class.feat.weights {model.coef_} + per.class.intercept {model.intercept_}"; 
 if 'coefs_' in dir(model): #mlp
  mw=f" per.feat.node.weights {model.coefs_} + per.node.intercept {model.intercepts_}"; 
 if 'support_vectors_' in dir(model): #svm #svmp
  mw=f" points of support vectors:\n{model.support_vectors_}"; 
 if 'feature_importances_' in dir(model) and not '-s.dt' in o: #rf ada, xgb
  mw=f" feature.importance:\n{model.feature_importances_}"; 
 if '.dt' in o:
  mwtree=SK.tree.export_text(model); mw=f" rule.tree:\n{mwtree}"; #dt
 if '.knn' in o:
  mw=model.kneighbors(x_); mw=f" per.instance.neighbors.indexes:\n{mw[1]}\n per.neighbor.distances:\n{mw[0]}"; #knn
 
 if 'get_params' in dir(model):
  print(model.get_params());
 if '.nn' in o:
  print(model.summary());
 if 'mw' in locals():
  print(mw);


#---evaluation

#if '-cv=' in o: #eval with cross validation (metrics: 'accuracy' 'balanced_accuracy')
# if target=='c':
#  scores = SK.model_selection.cross_val_score(model, x_, y_, scoring='balanced_accuracy', cv=cv, n_jobs=-1); print(f'eval with {folds}-fold cross-validation BAL ACC= %.3f (+ - %.2f)' % (NP.mean(scores), NP.std(scores))) 
# if target=='r': 
#  scores = SK.model_selection.cross_val_score(model, x_, y_, scoring='r2', cv=cv, n_jobs=-1); print('eval with {folds}-fold cross-validation R2= %.3f (+ - %.2f)' % (NP.mean(scores), NP.std(scores))) 


if '-s.' in o or '-t.' in o: #if the task is supervised run evaluation
 x_test=x_test2; #restore x_test in its dataframe form
 if target=='c': 
  #scores=model.fit(x_train, y_train);  
  #y_pred = model.predict_classes(x_test); print(scores); #print(y_test); print(y_pred);
  acc=SK.metrics.balanced_accuracy_score(y_test, y_pred); 
  print(f"eval predictions on test set. BAL ACC= {acc:.3f}"); 
  rr=SK.metrics.classification_report(y_test, y_pred); print(rr);
  cm=SK.metrics.confusion_matrix(y_test, y_pred, labels=classes); 
  cm=PD.DataFrame(cm); print("confusion matrix:\n",cm);
  af= open('log.txt', 'a'); 
  af.write(f"\n\n{f}, {o} -->BAL ACC= {acc:.3f}\n{rr}\nconfusion matrix:\n{cm}"); 
  af.close(); 
 if target=='r': 
  #scores=model.fit(x_train, y_train); y_pred=model.predict(x_test); 
  mae=SK.metrics.mean_absolute_error(y_test, y_pred); 
  y_scaled=SK.preprocessing.MinMaxScaler().fit_transform(y_test.to_numpy().reshape(-1,1)); 
  ys_=PD.DataFrame(y_scaled); #normalize ground truth
  p_scaled=SK.preprocessing.MinMaxScaler().fit_transform(y_pred.reshape(-1,1)); 
  ps_=PD.DataFrame(p_scaled); #normalize predictions
  r2=SK.metrics.r2_score(y_test, y_pred); #nmae=mean_absolute_percentage_error(y_test, y_pred); 
  print(f'eval on test set. R2= {r2:.3f}, MAE= {mae:.3f}'); #eval with train-test split. balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, mean_absolute_percentage_error
  NP.set_printoptions(precision=2); 
  if '-d.data' in o:
   print('predictions:'); 
   print(y_pred.flatten()); 
   print('ground truth:'); 
   print(y_test.to_numpy().flatten()); 
  af= open('log.txt', 'a'); 
  af.write(f"\n\n{f}, {o} --> R2= {r2:.3f}, MAE= {mae:.3f}\n"); 
  af.close();
  

#---visualizations
#print("\n");
if '-d.viz' in o:
 if '-s.' in o and nclass in locals(): #supervised scatterplots
  if task=='s' and target=='c': 
   projected = pca.fit_transform(x_test);
   MP.scatter(projected[:, 0], projected[:, 1], c=y_test, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', nclass))
   MP.xlabel('component 1');  MP.ylabel('component 2');  MP.colorbar();
   MP.title('2D ground truth test set'); MP.savefig(fname='test-space.png'); MP.show(); MP.clf();
   projected = pca.fit_transform(x_test);
   MP.scatter(projected[:, 0], projected[:, 1], c=y_pred, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', nclass))
   MP.xlabel('component 1');  MP.ylabel('component 2');  MP.colorbar();
   MP.title('2D predictions on test set'); MP.savefig(fname='testpred-space.png'); MP.show(); MP.clf();
  if task=='s' and target=='r':
   MP.scatter(ys_, ps_, alpha=0.8); MP.xlabel('ground truth (norm)');  
   MP.ylabel('predictions (norm)'); #MP.show(); MP.clf();
   MP.scatter(ys_, ys_, alpha=0.2); #MP.legend(handles=['ys_', 'ps_'], title='title', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small');
   MP.title('predictions-truth scatterplot'); MP.savefig(fname='target-pred-fit-space.png'); MP.show(); MP.clf();

 if '-t.' in o:# multiple timeseries line plots
  #y_pred=PD.DataFrame(y_pred);
  #print(dy_);print(y_test);print(y_pred);  print(len(dy_));print(len(y_test));print(len(y_pred));
  MP.plot(dy_, y_test, marker='', color='orange', linewidth=2, label='ground truth')
  MP.plot(dy_, y_pred, marker='', color='blue', linestyle='dashed', linewidth=2, label='predictions')
  MP.title('test set linechart'); MP.legend(); 
  MP.xticks(rotation=45, ha="right"); MP.show(); MP.clf();

 if '-x.' in o and 't_' in locals() and target=='c':#wordclouds
  print('terms most associated to each class')
  os.system("pip install wordcloud stop-words");
  from wordcloud import WordCloud;
  from collections import defaultdict
 # from stop_words import get_stop_words
  words_weights = defaultdict(list)
  for idx, class_ in enumerate(y_pred):
   words_weights[class_].append(orig_t_[idx])
 # stopwords = get_stop_words('it')
 # stopwords.extend(["http", "https"])
  for k, v in words_weights.items():
   MP.title('Class \'%s\'' % k);
   wc = WordCloud(background_color='white',  max_words=1000, colormap="berlin", contour_color="black");
   wc.generate(" ".join(v))
   MP.imshow(wc, interpolation='bilinear');
   MP.axis('off');
   MP.savefig('wordcloud_%s.png' % k); MP.show(); MP.clf();

timestamp=DT.datetime.now(); print(timestamp); 
print('---END PROCESS---'); sys.exit();
