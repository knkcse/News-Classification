import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time as tm
import string
from sklearn import svm
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix



def preProcessFunction(te,string):
	print "Pre processing  ",string 
	print "Word Tokenizations......."
	#tm.sleep(2)

	pre_Articles=[]
	for i in range(0,len(te['news'])):
		string=te['news'][i].decode('utf-8')
	
		Pre_X_Test=word_tokenize(string)
		pre_Articles.append(Pre_X_Test)
	stop_words = set(stopwords.words('english'))#setting stop words english
	print "Stop words removal....."
	#tm.sleep(2)
	stop_articles=[]
	for i in range(0,len(te['news'])):
		record=[]
		for w in pre_Articles[i]:
			if w not in stop_words:
				record.append(w)
		stop_articles.append(record)
	print "Lemmatizing...."
	#tm.sleep(2)
	lemmatizer = WordNetLemmatizer()

	lem_articles=[]
	for i in range(0,len(te['news'])):
		record=[]
		for w in stop_articles[i]:
				record.append(lemmatizer.lemmatize(w))
		lem_articles.append(record)
	#print "\n\n"
	#print lem_articles
	Panda_Test=pd.Series()
	for i in range(0,len(te['news'])):
		string=lem_articles[i]
		string=str(string).strip('[]')
		Panda_Test=Panda_Test.append(pd.Series(string))
	return Panda_Test







print "Reading training data file......"
#tm.sleep(3)
news_df = pd.read_csv("dataset.csv",error_bad_lines=False) # Train data set 
test_csv=news_df

print "Reading test data file....."
#tm.sleep(3)
te=pd.read_csv('Unique_Test_Articles.csv',error_bad_lines=False) # Test data 

categories={'type':{'business':1, 'entertainment':2, 'politics':3, 'sport':4, 'tech':5,'health':6,'others':7}}
news_df.replace(categories,inplace=True) # Representing categories as numbers
te.replace(categories,inplace=True)
print "No.of train data are ",len(news_df)
print "No. of test articles are ..",len(te['news'])
test_articles=preProcessFunction(te,"Test data")
news_df=preProcessFunction(news_df,"Train data")

X_train=news_df #Giving news columns for traing 
X_test=test_articles# test data news articles
y_train=test_csv['type']#Giving news categories for training
y_test=te['type']

print("Training dataset: ", len(X_train))
print("Test dataset: ", len(X_test))

#count_vector = CountVectorizer(stop_words='english',decode_error='ignore')
#training_data = count_vector.fit_transform(X_train)
#testing_data = count_vector.transform(X_test)

clf = svm.SVC(gamma=0.001, C=1000)

vect = TfidfVectorizer(stop_words='english',min_df=2)
training_data= vect.fit_transform(X_train)
testing_data=vect.transform(X_test)

print "Model "
#print clf
clf.fit(training_data,y_train)
pred=clf.predict(testing_data)
print pred
print "Prediction of news articles\n"
#print predictions
predictions=pd.DataFrame(columns=['news','category']) #initializing dataframe with columns
for i in range(0,len(te['news'])):
	#print te['news'][i],predictions[i]
	predictions= predictions.append({'news': te['news'][i],'category':pred[i]}, ignore_index=True)
print predictions

c_mat = confusion_matrix(y_test,pred)
#kappa = cohen_kappa_score(y_test,y_pred)
acc = accuracy_score(y_test,pred)
print "Confusion Matrix:\n", c_mat
##print "\nKappa: ",kappa
print"\nAccuracy: ",acc
