Topic Modeling and Sentiment Analysis of Customer reviews

The goal of this project was to:

1. Build a model that labels online reviews and puts them into specific catergories
2. Build a model that can predict the sentiment of an onile review

Use Cases:

1. Email filtering
2. Automatically Route support tickets to the correct department based on topic label
3. Sorting and labeling documents

Datasource: https://www.consumeraffairs.com/

Methods: 

BeautifulSoup- To scarpe data from consumeraffairs.com 
NLTK & gensim- To clean data and to perform some basic NLP
ELMO Vectors & Linear regression- Sentiment analysis
LDA Model- Topic Modeling
pylDavis: To create visual the topics

Models were deployed with Flask
