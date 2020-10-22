import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class DiseasePrediction:

    def _load_train_dataset(self):
        self.train_df = pd.read_csv('./dataset/disease_symptom_mapping.csv')
        cols = self.train_df.columns
        cols = cols[:-1] #except disease
        self.train_features = self.train_df[cols]
        self.train_labels = self.train_df['prognosis']
        
    def train_model(self):
        self._load_train_dataset()
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=0.33,
                                                          random_state=101)
        classifier = MultinomialNB()
        classifier = classifier.fit(X_train, y_train)
        confidence = classifier.score(X_val, y_val)

        y_pred = classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        score = cross_val_score(classifier, X_val, y_val, cv=3)
        print("Score ",score.mean())
        dump(classifier, str('./saved_model/' + "mnb" + ".joblib"))

    def make_prediction(self, test_data, saved_model_name=None):
        try:
            clf = load(str('./saved_model/'+ "mnb" + ".joblib"))
        except Exception as e:
            print("Model not found...")
        result = clf.predict(test_data)
        return result
        

    def symptomDetector(self,text):
        tokens = word_tokenize(text.lower())
        words = [word for word in tokens if word.isalpha()]
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        for s in stemmed:
            if s not in words:
                words.append(s)
        final_symps=list()
        symptoms_dataset = pd.read_csv('./dataset/Symptoms.csv')
        self.symptoms_list= symptoms_dataset.iloc[:-1,-1].tolist()
        for symp in self.symptoms_list:
            arr=symp.split("_")
            for i,v in enumerate(arr):
                arr[i]=v.strip()
            final_symps.append(arr)
        final_symps
        symp=list()
        for i,w in enumerate(words):
            for j,s in enumerate(final_symps):
                if(w==s[0]):
                    if len(s)>1:
                        word_index=i+1
                        c=0
                        for index,a in enumerate(s[1:]):
                            if s[index+1]==words[word_index]:
                                word_index+=1
                                c+=1
                        if c==len(s)-1:
                            z=w
                            for x in range(len(s)-1):
                                z=z+"_"+words[i+x+1]
                            symp.append(z)
                    else:
                        symp.append(words[i])
        return(symp)

    def inputNLP(self,symp):
        symptoms_dataset = pd.read_csv('./dataset/Symptoms.csv')
        self.symptoms_list= symptoms_dataset.iloc[:-1,-1].tolist()
        n=len(self.symptoms_list)
        final_input=[0 for i in range(n)]
        for s in symp:
            i=self.symptoms_list.index(s)
            final_input[i]=1
        return final_input



