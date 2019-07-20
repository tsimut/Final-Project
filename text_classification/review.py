from flask import Flask,render_template,url_for,request
import pandas as pd 
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import spacy

nlp= spacy.load('en_core_web_sm', disable=['parser', 'ner'])

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])    
def predict():
    if request.method == 'POST':
        message = request.form['message']
    data=[message]

    def sent_to_words(sentences):
        yield(gensim.utils.simple_preprocess(str(sentences), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))
    
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    
    

    data_words_nostops = remove_stopwords(data_words)
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print(data_lemmatized)
    
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    print(corpus)
    
    filename = 'final_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    scored=loaded_model[corpus]
   
    for topic in scored:
        prediction=(max(topic,key=lambda item:item[1])[0])
    
    if prediction== 0:
        topic="Credit Card"
    elif prediction== 1:
        topic="Interest Rate"
    elif prediction== 2:
        topic="Customer Service"
    else:
        topic="Bank Product"  

        
    return render_template('results.html',predict=topic)
     
if __name__ == '__main__':
	app.run(debug=True)