documentation='''
learnipy 
version 0.2, 2021
making machine learning and deep learning accessible to everyone
written with ♥ by Fabio Celli
email: fabio.celli.phd@gmail.com
twitter: @facells
tested in Google colab

===LICENSE===
Copyright (c) 2012-2021 Fabio Celli.
MIT license. 
Permissions: Commercial use,  Modification, Distribution, Private use
Limitations: Your liability, No warranty
Conditions: Report the following license and copyright notice with code.
"Permission is hereby granted, free of charge, to any person obtaining
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
 -d.gen=132  generate dataset, create gen.csv.params 1=num instances*1000, 3=num features*10, 2=num informative features*10
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
 -s.nn=f51   deep learning. params: f=feedfwd|i=imbalance|r=rnn|l=lstm|b=bilstm|g=gru|c=cnn. 5=2**units, 1=num layers
 ===evaluation===
 -e.tts=0.2  train-test split. params 0.2=20% test split. param not valid if test set is provided.
 '''

#TO DO: add drop features by index, gen markov chains, gen gan


import warnings; warnings.filterwarnings('ignore'); 
import pandas as PD;
import numpy as NP;
import tensorflow as TF;
import zipfile as ZF;
import sklearn as SK; from sklearn import *; from skimage.io import imread;
import matplotlib.pyplot as MP; MP.rcParams["figure.figsize"]=(4,3);
import gensim as W2V; from gensim.models.doc2vec import Doc2Vec, TaggedDocument;
import nltk; from nltk.tokenize import word_tokenize; nltk.download('punkt', quiet=True);
import sys; 
import re;
import os;
import math;
import joblib;

NP.random.seed(1); TF.random.set_seed(2);
TF.compat.v1.logging.set_verbosity(TF.compat.v1.logging.ERROR);

print('---START PROCESS---');

#---option reading
o=sys.argv[1]+' ';

if '-h' in o: 
 print(documentation); sys.exit();

#---data generation
if '-d.gen=' in o:
 r_=re.findall(r'-d.gen=(.+?) ',o); 
 ns=int(r_[0][0]);nf=int(r_[0][1]);ni=int(r_[0][2]); ns=ns*1000; nf=nf*10; ni=ni*10; nr=nf-ni; 
 print(f'generating dataset with {ns} samples and {nf} features, {ni} informative');
 x_, y_ = SK.datasets.make_classification(n_samples=ns, n_features=nf, n_informative=15, n_redundant=5, random_state=1); 
 x_=PD.DataFrame(x_); y_=PD.DataFrame(y_, columns=['class']); 
 g_=PD.concat([x_, y_], axis=1); g_.to_csv('gen.csv', sep=',', encoding='utf-8'); #generate a dummy dataset


#---files reading
try:
 f=sys.argv[2];
except:
 print('no file loaded'); sys.exit();


try:
 f2=sys.argv[3];
except:
 print('using training set');


#---data and model import
if 'f2' in locals(): #import csv test set
 if '.csv' in f2:
  testname=f2.replace('.csv', ''); datatype='csv';
  x2_=PD.read_csv(f2, sep=',', encoding='utf8'); 
  testinst=len(x2_.index); print('using training and test sets');
 if '.zip' in f2:
  testname=f2.replace('.zip', ''); datatype='zip';
  zip=ZF.ZipFile(f2, 'r'); i_=zip.namelist(); print('using training and test sets');
  if ',' in i_[0]: #extract supervised data from files in zip
   x2_=[]; y2_=[];
   for i in i_:
    l_=i.split(','); label=l_[1]; d=zip.open(i); #print(d);
    if '.jpg' in i or '.png' in i:
     d_=imread(d); dshape=d_.shape; d_=d_.flatten(); x2_.append(d_); y2_.append(label); #read and flatten images
 #ADD other file formats extraction
   x2_=NP.array(x2_).astype('float')/255.0;  y2_=NP.array(y2_).astype('int');
   x2_=PD.DataFrame(x2_); y2_=PD.Series(y2_); testinst=len(x2_.index); print(testinst);


if '.h4' in f and '-d.pred' in o: #import machine learning saved model
 loadmodel = joblib.load(f);
 o=f.replace('-',' -'); x_=x2_; #use model filename as o and test set as the main dataset to go into the pipeline with the same settings as the model trained
 if 'class' in x_:
  y_=x_['class']; x_=x_.drop(columns=['class']); task='s'; print('target found, suppose supervised task');
 if 'text' in x_:
  t_=x_['text']; x_=x_.drop(columns=['text']); 

if '.h5' in f and '-d.pred' in o: #import deep learning saved model
 loadmodel = TF.keras.models.load_model(f);
 o=f.replace('-',' -'); x_=x2_; #use model filename as o and test set as the main dataset to go into the pipeline with the same settings as the model trained
 if 'class' in x_:
  y_=x_['class']; x_=x_.drop(columns=['class']); task='s'; print('target found, suppose supervised task'); #TO CHECK task=='s'
 if 'text' in x_:
  t_=x_['text']; x_=x_.drop(columns=['text']); 


if '.csv' in f: #import .csv training set or all-in-one set
 print(f"processing {f} with {o}"); 
 filename=f.replace('.csv', ''); datatype='csv';
 x_=PD.read_csv(f, sep=',', encoding='utf8'); traininst=len(x_.index); dshape=[1];
 if '-p.rand' in o:
  x_ = x_.sample(frac=1).reset_index(drop=True); print('apply instance randomization in the training set'); #shuffle instances

 if 'x2_' in locals():
  tts=((100/(traininst+testinst))*testinst)/100; print(f'test set percentage={tts}');  #compute tts percentage  
  if not x2_.empty:
   x_=PD.concat([x_, x2_], axis=0, ignore_index=True); 

 if 'text' in x_:
  t_=x_['text']; x_=x_.drop(columns=['text']); 
 
 if 'class' in x_:
  y_=x_['class']; x_=x_.drop(columns=['class']); task='s'; print('target found, suppose supervised task');
 else:
  print('no class given. only unsupervised tasks enabled. for supervised tasks name "class" the target column in your data'); task='u';

if '.zip' in f: #extract data from .zip, loading in memory
 filename=f.replace('.zip', ''); datatype='zip';
 zip=ZF.ZipFile(f, 'r'); i_=zip.namelist(); 
 if ',' in i_[0]: #extract supervised data from files in zip
  task='s'; print('target found, suppose supervised task');
  x_=[]; y_=[];
  for i in i_:
   l_=i.split(','); label=l_[1]; d=zip.open(i); #print(d);
   if '.jpg' in i or '.png' in i:
    d_=imread(d); dshape=d_.shape; d_=d_.flatten(); x_.append(d_); y_.append(label); #read and flatten images
#ADD other file formats extraction
  x_=NP.array(x_).astype('float')/255.0;  y_=NP.array(y_).astype('int');
  x_=PD.DataFrame(x_); y_=PD.Series(y_); traininst=len(x_.index);
  if 'x2_' in locals():
   tts=((100/(traininst+testinst))*testinst)/100; print(f'test set percentage={tts}');  #compute tts percentage  
   x_=PD.concat([x_, x2_], axis=0, ignore_index=True); y_=PD.concat([y_, y2_], axis=0, ignore_index=True);

 else:  #extract unsupervised data from files in zip
  task='u'; print('no class given. only unsupervised tasks enabled. for supervised tasks format your data as name,label,.jpg');
  x_=[];
  for i in i_:
   d=zip.open(i); #print(d);
   if '.jpg' in i or '.png' in i:
    d_=imread(d); dshape=d_.shape; d_=d_.flatten(); x_.append(d_); #read and flatten images
#ADD other file formats extraction
  x_=NP.array(x_).astype('float')/255.0;
  x_=PD.DataFrame(x_);

 print(f"original data shape is {dshape}");


if not '.zip' in f and not '.csv' in f and not '.zip' in f2 and not '.csv' in f2:
 print('please input a .csv or .zip dataset'); sys.exit();

#---manage missin gvalues
if 'x_' in locals() and not '-u.apriori' in o:
 if x_.isnull().values.any():
  x_ = x_.fillna(0); print('filling missing values with 0 by default'); #fill all missing values in x_ with 0

if 'x_' in locals() and '-u.apriori' in o:
 if x_.isnull().values.any():
  x_ = x_.fillna(''); print('filling missing values with empty strings by default'); #fill all missing values in x_ with 0

if 'y_' in locals():
 if y_.isnull().values.any():
  y_ = y_.fillna(0); print('WARNING: there are missing values in the class, filled with 0. check if this compromise the results'); #fill all missing values in y_ with 0

if 't_' in locals():
 t_=t_.astype(str);

#---define target class type
if task=='s': 
 if y_.dtype.kind in 'biufc' and not '-d.c=l' in o:
  #y_=(y_-y_.min())/(y_.max()-y_.min()); 
  y_=y_.astype('float'); target='r'; print('read target as number, turned to float. to read target as a label use -d.c=l instead');
  if '-d.cnorm' in o:
   y_=(y_-y_.min())/(y_.max()-y_.min()); print('apply target numbers 0-1 normalization');
 else:
  y_=y_.astype('category').cat.codes.astype('int'); target='c'; print('read target as label, turned to integer category')


#---raw feature analysis

if '-u.arm' in o:
 x_=x_.to_numpy(); 
 x_ = NP.array([i for i in x_ if not '' in i]); #remove empty values from data
 from mlxtend.preprocessing import TransactionEncoder;
 import mlxtend;
 from mlxtend.frequent_patterns import apriori, association_rules;
 te = TransactionEncoder()
 te_ary = te.fit(x_).transform(x_)
 df = PD.DataFrame(te_ary, columns=te.columns_)

 frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
 #frequent_itemsets = fpmax(df, min_support=0.01, use_colnames=True)
 results=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7);
 af= open('analysis.txt', 'a'); af.write(results.to_string()+"\n\n"); af.close(); print(results);
 print('results printed in analysis.txt file');
 sys.exit();

#---data preprocessing 
if '-p.rand' in o:
 x_=SK.utils.shuffle(x_, random_state=1); print('apply instance position randomization'); #

#if '-p.stnd' in o: *TO REMOVE*
# x_scaled=SK.preprocessing.StandardScaler().fit_transform(x_.values); x_=PD.DataFrame(x_scaled, columns=x_.columns); print('apply feature standardization')

if '-p.tl' in o: 
 t_=t_.str.lower(); print('apply text to lowercase filter'); #string to lowercase 

if '-p.tc' in o:
 t_=t_.str.replace("\W", ' ', regex=True); t_=t_.str.replace(" {2,30}", ' ', regex=True); print('cleaning string from nonword characters and multiple spaces');

ncols=len(x_._get_numeric_data().columns); cols=len(x_.columns); #count of label and numeric columns in dataset

#---feature reduction
if '-r.svd=' in o: #define dimensions for SVD
 r_=re.findall(r'-r.svd=(.+?) ',o); svdim=int(r_[0]);
else:
 svdim=int(1+(cols/2)); o=f"-r.svd={svdim}"+o;

if '-r.lsa=' in o: #define dimensions for LSA
 r_=re.findall(r'-r.lsa=(.+?) ',o); lsadim=int(r_[0]); print(f'apply lsadim={lsadim}');
else:
 lsadim=50; o=f"-r.lsa={lsadim}"+o;


#---feature extraction
fx=0; #define flag for feature extraction
if not x_.empty and not 'zip' in datatype and cols >= ncols: #if data not empty, .csv and with label columns then extract features from labels, apply SVD
 x_=PD.get_dummies(x_); 
 print('async sparse one-hot matrix from labels:\n',x_) if '-d.data' in o else print('apply one-hot binarization of labels by default, obtain sparse async matrix'); #print(x_.describe()); 

 if len(x_.columns) >=2:
  svd=SK.decomposition.TruncatedSVD(svdim, random_state=1); x_=PD.DataFrame(svd.fit_transform(x_)); 
  print('sync dense SVD matrix from one-hot labels:\n',x_) if '-d.data' in o else print('apply Singular Value Decomposition of data by default, obtain dense sync matrix');
 else:
  x_=PD.DataFrame(); print('tabular data dropped because too small');


if 't_' in locals() and '-x.' in o: #extract features from text, apply LSA
 print('original text data:\n', t_) if '-d.data' in o else 0;

 if '-x.tm=' in o: #one hot word matrix
  r_=re.findall(r'-x.tm=(.+?) ', o); wu=int(r_[0]); 
  t = TF.keras.preprocessing.text.Tokenizer(num_words=wu, lower=True, char_level=False, oov_token=wu)
  t.fit_on_texts(t_); seq=t.texts_to_sequences(t_); wv_=t.sequences_to_matrix(seq, mode='freq');  t_=PD.DataFrame(wv_);
  fx=1; print('token freq matrix:\n',t_) if '-d.data' in o else print(f'extracting {wu} token frequence features from text');

 if '-x.ts=' in o: #one hot sequence matrix
  x_=PD.DataFrame(); print('tabular data dropped to prevent mixing tabular data and sequence text'); #empty x_ data to avoid mixing text and tabular data
  r_=re.findall(r'-x.ts=(.+?) ', o); wu=int(r_[0]); 
  t = TF.keras.preprocessing.text.Tokenizer(num_words=1000, lower=True, char_level=False, oov_token=0); t.fit_on_texts(t_);
  seq = t.texts_to_sequences(t_); seq=TF.keras.preprocessing.sequence.pad_sequences(seq, maxlen=wu); print('word indexes:\n',t.index_word); 
  t_=PD.DataFrame(seq);  #print(t_[0]);
  #t_=TF.one_hot(seq,wu); print('one hot seq:', t_);  dshape=(t_.shape[1],t_.shape[2]); t_=t_.numpy(); ft_=[]; [ft_.append(i.flatten()) for i in t_]; t_=PD.DataFrame(ft_);
  fx=1; print('token index sequences:\n',t_) if '-d.data' in o else print(f'extracting {wu} token indices sequence features from text');
  

 if '-x.ng=' in o:
  r_=re.findall(r'-x.ng=(.+?) ',o); mi=int(r_[0][0]); ma=int(r_[0][1]); ty=(r_[0][2]); mo=(r_[0][3]);
  if ty=='c' and mo=='f':
   w=SK.feature_extraction.text.CountVectorizer(ngram_range=(mi,ma),analyzer='char_wb',max_features=2000); wv_=w.fit_transform(t_); fx=1;
   fn_=[]; [fn_.append('t-'+i) for i in w.get_feature_names()]; t_=PD.DataFrame(wv_.toarray(), columns=fn_); 
   print('async sparse char ngram matrix:\n',t_) if '-d.data' in o else print(f'extract {mi}-{ma} char ngram from text');
  if ty=='w' and mo=='f':
   w=SK.feature_extraction.text.CountVectorizer(ngram_range=(mi,ma),analyzer='word',max_features=2000); wv_=w.fit_transform(t_); fx=1;
   fn_=[]; [fn_.append('t-'+i) for i in w.get_feature_names()]; t_=PD.DataFrame(wv_.toarray(), columns=fn_); 
   print('async sparse word ngram matrix:\n',t_) if '-d.data' in o else print(f'extract word {mi}-{ma}gram from text');
  if ty=='c' and mo=='t':
   w=SK.feature_extraction.text.TfidfVectorizer(ngram_range=(mi,ma),analyzer='char_wb',max_features=2000); wv_=w.fit_transform(t_); fx=1;
   fn_=[]; [fn_.append('t-'+i) for i in w.get_feature_names()]; t_=PD.DataFrame(wv_.toarray(), columns=fn_); 
   print('async sparse char ngram matrix:\n',t_) if '-d.data' in o else print(f'extract tf-idf {mi}-{ma} char ngram from text');
  if ty=='w' and mo=='t':
   w=SK.feature_extraction.text.TfidfVectorizer(ngram_range=(mi,ma),analyzer='word',max_features=2000); wv_=w.fit_transform(t_); fx=1;
   fn_=[]; [fn_.append('t-'+i) for i in w.get_feature_names()]; t_=PD.DataFrame(wv_.toarray(), columns=fn_); 
   print('async sparse word ngram matrix:\n',t_) if '-d.data' in o else print(f'extract tf-idf word {mi}-{ma}grams from text'); 
  #lsadim=int(len(fn_)/2);
  svd=SK.decomposition.TruncatedSVD(lsadim, random_state=1); t_=PD.DataFrame(svd.fit_transform(t_));
  print('sync dense LSA ngram matrix:\n',t_) if '-d.data' in o else print('apply LSA to ngrams by default, obtain dense sync matrix');
  if '-d.save ' in o:
   print(f"WARNING: the test set must contain at least {lsadim} instances for compatibility with the model"); 


 if '-x.w2v' in o: #word2vec
  print('extract word2vec dictionary from text. save w2v.txt, w2v.bin and w2v-space.png'); fx=1;
  t_=t_.str.split(pat=" "); wmodel=W2V.models.Word2Vec(t_, min_count=2); words=list(wmodel.wv.vocab);  
  wmodel.wv.save_word2vec_format('w2v.txt', binary=False); wmodel.save('w2v.bin'); #save word2vec dictionary
  X = wmodel[wmodel.wv.index2entity[:20]]; 
  pca=SK.decomposition.PCA(n_components=2); result=pca.fit_transform(X); MP.scatter(result[:, 0], result[:, 1]); 
  words_=list(wmodel.wv.index2entity[:20]); #print(words_); # fit a 2d PCA model to the w2v vectors
  [MP.annotate(word, xy=(result[i, 0], result[i, 1])) for i, word in enumerate(words_)]; 
  MP.title('w2v 2d space'); MP.savefig(fname='w2v-space'); MP.show(); MP.clf(); #visualize w2v-space and save it
  sys.exit();

 if '-x.d2v=' in o: #doc2vec
  r_=re.findall(r'-x.d2v=(.+?) ', o); size=int(r_[0]);  fx=1;
  t_=[TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(t_)]; 
  wmodel=Doc2Vec(vector_size=size); wmodel.build_vocab(t_); wmodel.train(t_, total_examples=len(t_), epochs=1); 
  t_=PD.DataFrame(wmodel.docvecs.vectors_docs); #turn to doc2vec vectors
  print('sync dense doc2vec matrix:\n',t_) if '-d.data' in o else print(f'extracting {size} doc2vec features from text');

#*TO REMOVE*
# if '-x.bert ' in o: #roBERTa
#  os.system('pip install transformers '); fx=1;
#  print('extracting 760 xlm-roBERTa features from text');
#  import torch; import transformers as PB; 
#  #size=t_.str.len().min(); t_=t_.str.slice(0, 50); print(t_);
#  m, t, w=(PB.XLMRobertaModel, PB.XLMRobertaTokenizerFast, 'xlm-roberta-base'); tokenizer = t.from_pretrained(w); wmodel=m.from_pretrained(w); 
#  tokenized = t_.apply((lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True))); max_len=0; padded = NP.array([i[0:5] for i in tokenized.values]); del(tokenizer);
#  attention_mask = NP.where(padded != 0, 1, 0);   input_ids = torch.tensor(padded); del(padded); attention_mask=torch.tensor(attention_mask); last_hidden_states = wmodel(input_ids, attention_mask=attention_mask); 
#  t_=PD.DataFrame(last_hidden_states[0][:,0,:].detach().numpy()); #get features from hidden states
 
 if '-x.bert ' in o: #bert uncased multi language
  os.system('pip install tensorflow-text'); fx=1;
  import shutil
  import tensorflow_hub as TH
  import tensorflow_text as text
  text_input = TF.keras.layers.Input(shape=(), dtype=TF.string)
  preprocessor = TH.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
  encoder_inputs = preprocessor(text_input)
  encoder = TH.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4", trainable=True)
  outputs = encoder(encoder_inputs);
  del(preprocessor); del(encoder);
  pooled_output = outputs["pooled_output"]      # [batch_size, 768].
  sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
  embedding_model = TF.keras.Model(text_input, pooled_output)
  sentences = TF.constant(t_); bert=embedding_model(sentences); t_=PD.DataFrame(bert.numpy()); 
  print('sync dense bert matrix:\n',t_) if '-d.data' in o else print(f'extracting 768 features with bert_multi_cased_L-12_H-768_A-12');

 if '-x.d=' in o: #user defined lexical resources
  fx=1;
  r_=re.findall(r'-x.d=(.+?) ', o); l=r_[0]; l_=open(l, encoding="utf8").read().splitlines(); print(f"using {l}");  h=l_.pop(0); h_=h.split(','); t2_=[]; #print(dir());
  for t in t_:
   t=' '+t+' '; c=t.count(' '); i_=[0 for i_ in range(len(h_))];
   for i in l_:
    w_=i.split(','); w=w_.pop(0);
    if re.search(w, t):
     for v in w_:
      i_[int(v)]=i_[int(v)]+1;  
   i_=[j/c for j in i_];
   t2_.append(i_);
  t_=PD.DataFrame(t2_,columns=h_); del(t2_);
  print('sync sparse matrix from lexicon:\n',t_) if '-d.data' in o else print(f'extracting features with {l}');

 
#---data aggregation
 if fx==1: #if feature extraction performed concat x_ and t_, else drop t_
  x_=PD.concat([x_, t_], axis=1); 
 else:
  print('no text feature extraction. text column dropped');
  if x_.empty:
   print('no features. prcess stopped'); sys.exit();

inst=len(x_.index); feat=len(x_.columns); print(f'dataset shape: {inst} instances, {feat} features');


#---class statistics
if '-d.viz' in o:
 if task=='s':
  MP.hist(y_, color='black', edgecolor='black', linewidth=0);  MP.ylabel('frequency'); MP.title('class distribution');  MP.savefig(fname='class-dist'); MP.show(); MP.clf(); #class dist
if target=='r':
 mu=y_.mean(); mi=y_.min(); ma=y_.max(); sd=y_.std(); me=y_.median(); print(f"min={mi:.1f} max={ma:.1f} avg={mu:.3f} sd={sd:.3f} med={me:.3f} numeric target distribution"); 
 nclass=1;
if target=='c':
 print(f"class freq"); print(y_.value_counts()); 
 ynp_=y_.to_numpy(); classes, counts=NP.unique(ynp_, return_counts=True); nclass=len(classes); print(f"num classes= {nclass}")


#---processed features unsupervised learning: analysis and clustering
xc_=PD.concat([y_, x_], axis=1); xcc=xc_.corr(); xcc=PD.Series(xcc.iloc[0]); xcc=xcc.iloc[1:]; 
complexity=1-(xcc.abs().max()); print(f"corr. complexity= {complexity:.3f}"); #compute complexity
if '-u.corr' in o: #correlation analysis 
 af= open('analysis.txt', 'a'); af.write("correlation matrix:\n\n"+xc_.corr().to_string()+"\n\n"); af.close();

if '-u.km=' in o: #kmeans clustering
 r_=re.findall(r'-u.km=(.+?) ',o); nk=int(r_[0]);
 clust = SK.cluster.KMeans(n_clusters=nk, random_state=0).fit(x_); l_=PD.DataFrame(clust.labels_); l_.columns=['kmeans']; x_=PD.concat([x_,l_], axis=1); g_=x_.groupby('kmeans').mean(); print('applied kmeans clustering. added 1 feature'); 
 af= open('analysis.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close();  print(f"num clusters= {nk}");
 if '-d.viz' in o:
  pca=SK.decomposition.PCA(2); projected=pca.fit_transform(x_); MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); MP.title('2D PCA data space kmeans clusters'); MP.savefig(fname='pca-cluster-space.png'); MP.show(); MP.clf(); #pca space

if '-u.optics' in o: #optics clustering
 clust = SK.cluster.OPTICS(min_cluster_size=None).fit(x_); l_=PD.DataFrame(clust.labels_); l_.columns=['optics']; x_=PD.concat([x_,l_], axis=1); g_=x_.groupby('optics').mean(); print('applied optics clustering. added 1 feature');
 af= open('analysis.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); ynp_=l_.to_numpy(); classes, counts=NP.unique(ynp_, return_counts=True); nk=len(classes); print(f"num clusters= {nk}");
 if '-d.viz' in o:
  pca=SK.decomposition.PCA(2); projected=pca.fit_transform(x_); MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); MP.title('2D PCA data space optics clusters'); MP.savefig(fname='pca-cluster-space.png'); MP.show(); MP.clf(); #pca space

if '-u.msh' in o: #meanshift clustering
 clust = SK.cluster.MeanShift().fit(x_); l_=PD.DataFrame(clust.labels_); l_.columns=['mshift']; x_=PD.concat([x_,l_], axis=1); g_=x_.groupby('mshift').mean(); print('applied mshift clustering. added 1 feature');
 af= open('analysis.txt', 'a'); af.write(g_.to_string()+"\n\n"); af.close(); ynp_=l_.to_numpy(); classes, counts=NP.unique(ynp_, return_counts=True); nk=len(classes); print(f"num clusters= {nk}");
 if '-d.viz' in o:
  pca=SK.decomposition.PCA(2); projected=pca.fit_transform(x_); MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('brg', nk));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); MP.title('2D PCA data space meanshift clusters'); MP.savefig(fname='pca-cluster-space.png'); MP.show(); MP.clf(); #pca space



#---preprocessing features
if not '-x.ts=' in o or not 't_' in locals():
 maxval=2;
 if '-p.norm' in o or '-s.sgd' in o or '-s.nb' in o or '-s.nn' in o: #column normalization range 0-1 (required for sgd and nb)
  x_scaled=SK.preprocessing.MinMaxScaler().fit_transform(x_.values); x_=PD.DataFrame(x_scaled, columns=x_.columns); print('apply feature normalization'); #normalization by column
if '-x.ts=' in o:
 #x_=x_.div(x_.sum(axis=1), axis=0); print('apply instance normalization'); #normalize by row
 #x_scaled=SK.preprocessing.MinMaxScaler().fit_transform(x_.values); x_=PD.DataFrame(x_scaled, columns=x_.columns); print('apply feature normalization'); #normalization by column
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
x_=PD.DataFrame(x_.values.astype(NP.float64), columns=x_.columns); print('turn all features to float numbers.'); #print(x_);
#print(list(x_)); #print(x_.dtypes);
sparseness=0; sparsemsg=''; sparse = sum((x_ == 0).astype(int).sum())/x_.size; #initialize sparseness check
if sparse>=0.9:
 sparseness=1; sparsemsg='warning: high data sparseness, better to apply feature reduction. ';
print(f"sparseness= {sparse:.3f}");


x_.columns = range(x_.shape[1]); #duplicate_columns = x_.columns[x_.columns.duplicated()]; print(duplicate_columns);#make columns unique 


if '-d.fdist' in o:
 #print(x_.dtypes); print(y_.dtypes);
 for col in x_:
  mu=x_[col].mean(); mi=x_[col].min(); ma=x_[col].max(); me=x_[col].median(); sd=x_[col].std(); print(f"min={mi:.1f} max={ma:.1f} avg={mu:.3f} sd={sd:.3f} med={me:.3f}  distribution of feature {col}"); 

if '-d.data' in o:
 print('data ready for learning:'); print(x_);

if '-d.viz' in o:
 if task=='s':
  pca=SK.decomposition.PCA(2); projected=pca.fit_transform(x_); MP.scatter(projected[:, 0], projected[:, 1], c=y_, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', nclass));
  MP.xlabel('component 1'); MP.ylabel('component 2');
  if target=='c':
   MP.colorbar(); 
  MP.title('2D PCA data space and classes'); MP.savefig(fname='pca-data-space.png'); MP.show(); MP.clf(); #pca space
 else:
  pca=SK.decomposition.PCA(2); projected=pca.fit_transform(x_); MP.scatter(projected[:, 0], projected[:, 1], c=y_, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', 1));
  MP.xlabel('component 1'); MP.ylabel('component 2'); MP.colorbar(); MP.title('2D PCA unsupervised data space'); MP.savefig(fname='pca-data-space.png'); MP.show(); MP.clf(); #pca space



#---predictions with saved model
if '.h4' in f: #apply machine learning saved model
 #loadmodel = joblib.load(f);
 y2_pred=loadmodel.predict(x_); y2_pred=PD.DataFrame(y2_pred); #print(type())
 n_=PD.concat([x2_,y2_pred], axis=1); print('applied saved model on supplied test set'); 
 af= open(f"{testname}-predictions.csv", 'w'); af.write(n_.to_csv()); af.close();  print(f"data with predictions saved as {testname}-predictions.csv");
 sys.exit();

if '.h5' in f: #apply deep learning saved model
 y2_pred=loadmodel.predict(x_); 
 if target=='c':
  y2_pred=PD.DataFrame(y2_pred); y2_pred=y2_pred.idxmax(axis=1); #print(y2_pred);
 if target=='r':
  y2_pred=PD.DataFrame(y2_pred);
 n_=PD.concat([x2_,y2_pred], axis=1); print('applied saved model on supplied test set'); 
 af= open(f"{testname}-predictions.csv", 'w'); af.write(n_.to_csv()); af.close();  print(f"data with predictions saved as {testname}-predictions.csv");
 sys.exit();


#---evaluation method
#if '-e.cv=' in o: # cross validation *TO REMOVE*
# r_=re.findall(r'-e.cv=(.+?) ',o); folds=int(r_[0]); cv = SK.model_selection.RepeatedKFold(n_splits=folds, random_state=1); # prepare the cross-validation eval

if '-e.tts' in o:
 r_=re.findall(r'-e.tts=(.+?) ',o); mi=float(r_[0]); 
 if 'x2_' in locals():
  mi=tts;
else:
  mi=0.3; print('using 70% train 30% test percentage split by default');
if 'mi' in locals(): #if split percentage is defined, then split train and test sets
 x_train, x_test, y_train, y_test = SK.model_selection.train_test_split(x_, y_, test_size=mi, shuffle=False, stratify=None, random_state=1) # prepare train test eval 
 xtrain_inst=len(x_train.index); feat=len(x_train.columns); print(f'training set shape: {xtrain_inst} instances, {feat} features');
 xtest_inst=len(x_test.index); feat=len(x_test.columns); print(f'test set shape: {xtest_inst} instances, {feat} features');
 x_test2=x_test; #create a copy of x_test for evaluationin case of shape change



#---supervised learning
if '-s.base' in o:
 model=SK.dummy.DummyClassifier(strategy='most_frequent'); print('compute majority baseline'); model.fit(x_train, y_train); y_pred=model.predict(x_test);  #'most_frequent', 'prior',
 
if '-s.nb' in o and target=='c':
 model=SK.naive_bayes.ComplementNB();model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply complement naive bayes classification (on normalized space)');
if '-s.nb' in o and target=='r':
 model=SK.linear_model.BayesianRidge();model.fit(x_train, y_train); y_pred=model.predict(x_test); y_pred=y_pred.flatten(); print('apply bayesian ridge regression (on normalized space)');

if '-s.lcm' in o and target=='c':
 model=SK.discriminant_analysis.LinearDiscriminantAnalysis(); model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply LinearDiscriminantAnalysis classification');
if '-s.lcm' in o and target=='r':
 model=SK.cross_decomposition.PLSRegression(max_iter=500);model.fit(x_train, y_train); y_pred=model.predict(x_test); y_pred=y_pred.flatten(); print('apply PartialLeastSquare regression');

if '-s.lr' in o and target=='c':
 model=SK.linear_model.LogisticRegression(max_iter=5000);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply logistic regression classification');
if '-s.lr' in o and target=='r':
 model=SK.linear_model.LinearRegression();model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply linear regression');

if '-s.sgd' in o and target=='c':
 model=SK.linear_model.SGDClassifier(shuffle=False);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply stochastic gradient descent classification (on normalized space)');
if '-s.sgd' in o and target=='r':
 model=SK.linear_model.SGDRegressor(shuffle=False);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply sochastic gradient descent regression (on normalized space)');

if '-s.knn' in o and target=='c':
 model=SK.neighbors.KNeighborsClassifier();model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply k nearest neighbors classification');
if '-s.knn' in o and target=='r':
 model=SK.neighbors.KNeighborsRegressor();model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply k nearest neighbors regression');

if '-s.mlp' in o and target=='c':
 model=SK.neural_network.MLPClassifier(random_state=1);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply multi layer perceptron classification');
if '-s.mlp' in o and target=='r':
 model=SK.neural_network.MLPRegressor(random_state=1);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply multi layer perceprtron regression');

if '-s.svmp=' in o and target=='c':
 r_=re.findall(r'-s.svmp=(.+?) ',o); mi=int(r_[0]); model=SK.svm.NuSVC(kernel='poly', degree=mi);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply support vector machines with polynomial kernel');
if '-s.svmp=' in o and target=='r':
 r_=re.findall(r'-s.svmp=(.+?) ',o); mi=int(r_[0]); model=SK.svm.NuSVR(kernel='poly', degree=mi);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply support vector machines with polynomial kernel');

if '-s.svm' in o and target=='c':
 model=SK.svm.NuSVC(kernel='rbf', nu=0.5);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply support vector machines with rbf kernel');
if '-s.svm' in o and target=='r':
 model=SK.svm.NuSVR(kernel='rbf', nu=0.5);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply support vector machines with rbf kernel');

if '-s.rf' in o and target=='c':
 model=SK.ensemble.RandomForestClassifier(random_state=1);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply random forest classification');
if '-s.rf' in o and target=='r':
 model=SK.ensemble.RandomForestRegressor(random_state=1);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply random forest regression');

if '-s.ada' in o and target=='r':
 model=SK.ensemble.AdaBoostRegressor(random_state=1);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply adaboost regression')
if '-s.ada' in o and target=='c':
 model=SK.ensemble.AdaBoostClassifier(random_state=1);model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply adaboost classification');

if '-s.dt' in o and target=='c':
 model=SK.tree.DecisionTreeClassifier();model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply decision trees classification, get rules:'); rules=SK.tree.export_text(model);print(rules);
if '-s.dt' in o and target=='r':
 model=SK.tree.DecisionTreeRegressor();model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply decision trees regression. get rules:'); rules=SK.tree.export_text(model);print(rules);

if '-s.xgb' in o and target=='c':
 import xgboost; model=xgboost.XGBClassifier();model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply xgboost classification');
if '-s.xgb' in o and target=='r':
 import xgboost; model=xgboost.XGBRegressor();model.fit(x_train, y_train); y_pred=model.predict(x_test); print('apply xgboost regression');


if 'model' in locals():
 if target=='c': #compute classes for machine learning
  classes=model.classes_;
 if '-d.save' in o and not '-s.nn' in o: #save machine learning models
  opt=re.sub(r'-d.save|-e\..+|-d.viz| ','', o); 
  if target=='c':
   opt='-d.c=l'+opt;
  else:
   opt='-d.c=0'+opt;
  joblib.dump(model, f"{filename}{opt}.h4");print(f"model saved as {filename}{opt}.h4");
  #af= open(f"{filename}-format4model.csv", 'w'); af.write(x_test.to_csv()); af.close();  




if '-s.nn' in o:
 x_train=x_train.to_numpy(); x_test=x_test.to_numpy(); y_train=y_train.to_numpy(); y_test=y_test.to_numpy(); #turn dataframe to numpy

 #l2=2; nu=5; 
 nl=int((complexity/2)*10); nu=int(math.sqrt(feat * nclass)); #automatic selection of num. layers and nodes

 r_=re.findall(r'-s.nn=(.+?) ',o); 
 if len(r_[0]) > 1:
  nt=r_[0][0]; nu=int(r_[0][1]); nl=int(r_[0][2]);  
  print('using neural network options')
 else:
  print('no (enough) neural network options given. using default settings.');

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

 print(f"layer cofig: actv={activ}, nodes={nu}"); 
 print(f"network config: optimizer={opt}, out_actv={outactiv}");
 print(f"validator config: loss={los}, metric={metric},");
 #optimizers: adam(robust) sgd(fast) rmsprop(variable) adadelta(for sparse data)
 #activations: linear(lin,-inf/+inf) gelu(nonlin,-0.17/+inf) selu(nonlin,-a/+inf) sigmoid(nonlin,0/1) tanh(nonlin,-1/+1) softplus(nonlin,0/+inf) softsign(nonlin,-1/+1). linear=regression, relu|selu|gelu=general purpose, sigmoid|tanh=binary classification, softplus|softsign=multiclass classifiction
 #losses: hinge kld cosine_similarity mse msle huber binary_crossentropy sparse_categorical_crossentropy
 #metrics: mape mae accuracy top_k_categorical_accuracy categorical_accuracy
 if '-s.nn=f' in o: #feedforward
  model=TF.keras.Sequential();
  model.add(TF.keras.layers.Dense(feat, activation=activ)); #initial nodes are=num features
  model.add(TF.keras.layers.Dense(nu*2, activation=activ));
  [model.add(TF.keras.layers.Dense(nu, activation=activ)) for n in range(0,nl)]
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
  if len(dshape)==1:
   model = TF.keras.Sequential();
   model.add(TF.keras.layers.Embedding(maxval,nu,input_length=feat))
   [model.add(TF.keras.layers.Conv1D(int(nu/2), 7, activation=activ,padding='same' )) for n in range(0,nl)]
   #model.add(TF.keras.layers.Conv1D(32, 7, activation=activ,padding='same')); 
   model.add(TF.keras.layers.GlobalMaxPooling1D());
   model.add(TF.keras.layers.Dropout(0.3));
   model.add(TF.keras.layers.Dense(nclass, activation=outactiv));
  if len(dshape)==2:
   x_train=x_train.reshape((xtrain_inst,dshape[0],dshape[1],1));
   x_test=x_test.reshape((xtest_inst,dshape[0],dshape[1],1));
   model = TF.keras.Sequential();
   [model.add(TF.keras.layers.Conv2D(nu, kernel_size=(3, 3), activation=activ, input_shape=(dshape[0],dshape[1],1))) for n in range(0,nl)]
  # [model.add(TF.keras.layers.Conv2D(nu, kernel_size=(3, 3), activation=activ)) 
   model.add(TF.keras.layers.MaxPooling2D(pool_size=(2, 2)));
   model.add(TF.keras.layers.Conv2D(nu*2, kernel_size=(3, 3), activation=activ));
   model.add(TF.keras.layers.MaxPooling2D(pool_size=(2, 2)));
   model.add(TF.keras.layers.Flatten()); model.add(TF.keras.layers.Dropout(0.5));
   model.add(TF.keras.layers.Dense(nclass, activation=outactiv));

 
 model.compile(optimizer=opt,  loss=los,  metrics=[metric]);

 print('create models on training set. max 100 epochs, stop after 5 epochs with no improvement');
 earlystop = TF.keras.callbacks.EarlyStopping(patience=5, monitor=metric); model.fit(x_train, y_train, epochs=100, callbacks=[earlystop], verbose=1);# 
 print('evaluate model on test set'); model.evaluate(x_test,  y_test, verbose=2); 
 print(model.summary());
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
   opt='-d.c=l'+opt;
  if target=='r':
   opt=f"-d.c=0"+opt;
  model.save(f"{filename}{opt}.h5"); print(f"model saved as {filename}{opt}.h5");


if not 'y_pred' in locals():
 print('no supervised model trained. process stopped'); sys.exit();


#---evaluation

#if '-cv=' in o: #eval with cross validation (metrics: 'accuracy' 'balanced_accuracy')
# if target=='c':
#  scores = SK.model_selection.cross_val_score(model, x_, y_, scoring='balanced_accuracy', cv=cv, n_jobs=-1); print(f'eval with {folds}-fold cross-validation BAL ACC= %.3f (+ - %.2f)' % (NP.mean(scores), NP.std(scores))) 
# if target=='r': 
#  scores = SK.model_selection.cross_val_score(model, x_, y_, scoring='r2', cv=cv, n_jobs=-1); print('eval with {folds}-fold cross-validation R2= %.3f (+ - %.2f)' % (NP.mean(scores), NP.std(scores))) 

if '-s.' in o: #if the task is supervised run evaluation
 x_test=x_test2; #restore x_test in its dataframe form
 if target=='c': 
  #scores=model.fit(x_train, y_train);  y_pred = model.predict_classes(x_test); print(scores); #print(y_test); print(y_pred);
  acc=SK.metrics.balanced_accuracy_score(y_test, y_pred); 
  print(f"eval predictions on test set. BAL ACC= {acc:.3f}"); 
  rr=SK.metrics.classification_report(y_test, y_pred); print(rr);
  cm=SK.metrics.confusion_matrix(y_test, y_pred, labels=classes); cm=PD.DataFrame(cm); print("confusion matrix:\n",cm);
  af= open('results.txt', 'a'); af.write(f"\n\n{f}, {o} -->BAL ACC= {acc:.3f}\n{rr}\nconfusion matrix:\n{cm}"); af.close(); 
 if target=='r': 
  #scores=model.fit(x_train, y_train); y_pred=model.predict(x_test); 
  y_scaled=SK.preprocessing.MinMaxScaler().fit_transform(y_test.to_numpy().reshape(-1,1)); ys_=PD.DataFrame(y_scaled); #normalize ground truth
  p_scaled=SK.preprocessing.MinMaxScaler().fit_transform(y_pred.reshape(-1,1)); ps_=PD.DataFrame(p_scaled); #normalize predictions
  r2=SK.metrics.r2_score(y_test, y_pred); nmae=SK.metrics.mean_absolute_error(ys_, ps_); 
  print(f'eval on test set. R2= {r2:.3f}, NORM MAE= {nmae:.3f}'); #eval with train-test split. balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, mean_absolute_percentage_error
  NP.set_printoptions(precision=2); print('predictions:'); print(y_pred); print('ground truth:'); print(y_test.to_numpy()); 
  af= open('results.txt', 'a'); af.write(f"\n\n{f}, {o} --> R2= {r2:.3f}, NORM MAE= {nmae:.3f}\n"); af.close();



#---visualizations
#print("\n");
if '-d.viz' in o:
 if task=='s' and target=='c': 
  pca = SK.decomposition.PCA(2); projected = pca.fit_transform(x_test);
  MP.scatter(projected[:, 0], projected[:, 1], c=y_test, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', nclass))
  MP.xlabel('component 1');  MP.ylabel('component 2');  MP.colorbar();
  MP.title('2D PCA ground truth test set'); MP.savefig(fname='pca-test-space.png'); MP.show(); MP.clf();

  pca = SK.decomposition.PCA(2); projected = pca.fit_transform(x_test);
  MP.scatter(projected[:, 0], projected[:, 1], c=y_pred, edgecolor='none', alpha=0.8, cmap=MP.cm.get_cmap('copper_r', nclass))
  MP.xlabel('component 1');  MP.ylabel('component 2');  MP.colorbar();
  MP.title('2D PCA predictions on test set'); MP.savefig(fname='pca-testpred-space.png'); MP.show(); MP.clf();
 if task=='s' and target=='r':
  MP.scatter(ys_, ps_, alpha=0.8); MP.xlabel('target ground truth');  MP.ylabel('target predictions'); #MP.show(); MP.clf();
  MP.scatter(ys_, ys_, alpha=0.2); #MP.legend(handles=['ys_', 'ps_'], title='title', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small');
  MP.title('target-predictions fit'); MP.savefig(fname='target-pred-fit-space.png'); MP.show(); MP.clf();

