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
import time as tm
import string
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def preProcessFunction(te,string):
	print "Pre processing  ",string 
	print "Word Tokenizations......."
	tm.sleep(2)

	pre_Articles=[]
	for i in range(0,len(te['news'])):
		string=te['news'][i].decode('utf-8')
	
		Pre_X_Test=word_tokenize(string)
		pre_Articles.append(Pre_X_Test)
	stop_words = set(stopwords.words('english'))#setting stop words english
	print "Stop words removal....."
	tm.sleep(2)
	stop_articles=[]
	for i in range(0,len(te['news'])):
		record=[]
		for w in pre_Articles[i]:
			if w not in stop_words:
				record.append(w)
		stop_articles.append(record)
	print "Lemmatizing...."
	tm.sleep(2)
	lemmatizer = WordNetLemmatizer()

	lem_articles=[]
	for i in range(0,len(te['news'])):
		record=[]
		for w in stop_articles[i]:
				record.append(lemmatizer.lemmatize(w))
		lem_articles.append(record)
	#print "\n\n"
	#print lem_articles[0]
	Panda_Test=pd.Series()
	for i in range(0,len(te['news'])):
		string=lem_articles[i]
		string=str(string).strip('[]')
		Panda_Test=Panda_Test.append(pd.Series(string))
	return Panda_Test




print "Reading training data file......"
tm.sleep(3)
news_df = pd.read_csv("dataset.csv",error_bad_lines=False) # Train data set 
test_csv=news_df

print "Reading test data file....."
tm.sleep(3)
te=pd.read_csv('Unique_Test_Articles.csv',error_bad_lines=False) # Test data 

categories={'type':{'business':1, 'entertainment':2, 'politics':3, 'sport':4, 'tech':5,'health':6,'others':7}}
news_df.replace(categories,inplace=True) # Representing categories as numbers
te.replace(categories,inplace=True)
print "No.of train data are ",len(news_df)
print "No. of test articles are ..",len(te['news'])
test_articles=preProcessFunction(te,"Test data")
news_df=preProcessFunction(news_df,"Train data")

#print test_articles[0]
X_train=news_df #Giving news columns for traing 
X_test=test_articles# test data news articles
y_train=test_csv['type']#Giving news categories for training
y_test=te['type']

print "Making count vectorizer for test data and traing data.....\n"
tm.sleep(2)
count_vector = CountVectorizer()

#Below code is to convert list to pandas object so that countVectorizer will be used without errors


training_data = count_vector.fit_transform(X_train)
print training_data.toarray()
print "Total features extracte ",len(count_vector.get_feature_names())# will give all features from train data
testing_data = count_vector.transform(test_articles)
print testing_data
print "Model..."
tm.sleep(2)
naive_bayes = MultinomialNB()
print "Training the model....."
tm.sleep(2)
print naive_bayes.fit(training_data, y_train)
print "Testing the model...."
tm.sleep(2)
#print (naive_bayes.predict_proba(testing_data))# Gives the probabilities of test vector
predictions = naive_bayes.predict(testing_data)
print "Prediction of news articles\n"
#print predictions
pred=pd.DataFrame(columns=['news','category']) #initializing dataframe with columns
for i in range(0,len(te['news'])):
	#print te['news'][i],predictions[i]
	pred= pred.append({'news': te['news'][i],'category':predictions[i]}, ignore_index=True)
print pred


c_mat = confusion_matrix(y_test,predictions)

print c_mat
print "\n\n"

print("Accuracy score: ", accuracy_score(y_test, predictions))


#string=X_test[0]
##print string
#string=str(string).strip('[]')
#string2=pd.Series(string)
#str1=X_test[1]
#str2=str(str1).strip('[]')
#string2=string2.append(pd.Series(str2))

#print "Here it is "
#print string2
#testing_data = count_vector.transform(string2)
##print count_vector.get_feature_names()
##print "Naveen Kumar"
##print count_vector.inverse_transform(testing_data)
##print "Naveen",count_vector.inverse_transform(testing_data[0])
##print "\nTransform\n"
##test=count_vector.transform(["How are you man?","Im good how are you"]) # it counts the words which were fixed by fit_transform
##print test.toarray()
##print test
##print "Naveen\n",count_vector.inverse_transform(test[0])
##print count_vector.get_feature_names()


#naive_bayes = MultinomialNB()
#naive_bayes.fit(training_data, y_train)
#predictions = naive_bayes.predict(testing_data)
#print "Prediction of news articles\n"
#print predictions
##pred=pd.DataFrame(columns=['news','category'])
##print len(te['news'])
##for i in range(0,len(te['news'])):
##	#print te['news'][i],predictions[i]
##	pred= pred.append({'news': te['news'][i],'category':predictions[i]}, ignore_index=True)
##print pred
##print "\n\n"
##c_mat = confusion_matrix(y_test,predictions)
#print c_mat
#print "\n\n"
#print("Accuracy score: ", accuracy_score(y_test, predictions))
#print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
#print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
#print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))
