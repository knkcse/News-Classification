import newspaper
import urllib2
import csv
import pandas as pd
from bs4 import BeautifulSoup
from newspaper import Article
try:
	types=['business','tech','entertainment','sport','health','tech'] #categories
	websites=["/BUSINESS.en_in/Business?ned=in&hl=en-IN&gl=IN","/TECHNOLOGY.en_in/Technology?ned=in&hl=en-IN&gl=IN","/ENTERTAINMENT.en_in/Entertainment?ned=in&hl=en-IN&gl=IN","/SPORTS.en_in/Sport?ned=in&hl=en-IN&gl=IN","/HEALTH.en_in/Health?ned=in&gl=IN&hl=en-IN","/SCIENCE.en_in/Science?ned=in&gl=IN&hl=en-IN"]
	url="https://news.google.com/news/headlines/section/topic"
	news=[] #to store articles news
	category=[]
	urls_c=[]
	for i in range (0,6):
		try:
			page= urllib2.urlopen(url+websites[i])# Complete url of google news
			#urllib2 is used to open urls and mainly focuses on scrawling the web page from url
			
			soup = BeautifulSoup(page,'html.parser')#Making beautiful soup object for the page
			#BeautifulSoup will not fetch the page it pulls the data from fetched html/xml page. It creates a parse tree for the page which will be used to pull 				the required data.It provides navigating,searching and modification of of parse html tree.
			body=soup.find(class_='KaRWed XogBlf')#Finding required part of the html page. (What we need)
			#print body
			#title_soup = soup.findAll('span', 'titletext') 
			#link_soup  = soup.findAll('a', 'article') 
			#print title_soup
			#print body
			#i=0;
	
			j=0
			print types[i]+" starts here\n"
			for link in body.find_all('a',{"target":"_blank"}):   #Finding articles urls and getting information from the url's page using newspaper
				print(link.get('href'))
				#if(j==15): break
				try:
					article_url=link.get('href')#Taking href of anchor tag
					article_page=Article(article_url)#Initializing specific news article by using Article
					#print article_page
					
					article_page.download()
					print article_page
					article_page.parse()
					#print(article_page.text)
					data=article_page.text
					if(len(data)>250):
						news.append(data)
						category.append(types[i])
						urls_c.append(link.get('href'));
					
				except:
					print "Unable to fetch this article"
				j+=1
		except:
			print "Unable to read url content"
	data = {'news': news, 'type': category,'url':urls_c}       
	df = pd.DataFrame(data)
	#print df
	print 'writing csv flie ...'
	df.to_csv('Articles.csv', index=False,encoding="utf-8")

	

	#To remove duplicates 
	
	df = pd.read_csv('Articles.csv') #Reading csv file and storing the data into pandas object (DataFrame)
	df.drop_duplicates(subset=None, inplace=True)# Function to remove duplicates
	df.to_csv('UniqueArticles.csv',index=False,encoding='utf-8')# Writing to another csv file



	#To remove null articles
	obj=pd.read_csv('UniqueArticles.csv')
	#print obj
	#print obj.shape
	#print obj.isnull().sum()
	obj=obj.dropna()
	#print obj
	#print obj.shape
	#df.to_csv('Articles.csv', index=False,encoding="utf-8")
	obj.to_csv('Unique_Test_Articles.csv',index=False,encoding='utf-8')
		
except:
	print "Url reading problem"	

print "Unique test articles  collections is completed"



import NaiveBayes










