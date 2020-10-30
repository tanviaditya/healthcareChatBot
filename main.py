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
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pickle
from sklearn.naive_bayes import MultinomialNB,GaussianNB

class DiseasePrediction:

    def _load_train_dataset(self):
        self.train_df = pd.read_csv('./dataset/disease_symptom_mapping.csv')
        cols = self.train_df.columns
        cols = cols[:-1] #except disease
        self.train_features = self.train_df[cols]
        self.train_labels = self.train_df['prognosis']
        
    def train_model(self):
        self._load_train_dataset()
        X_train, X_val, y_train, y_test = train_test_split(self.train_features, self.train_labels,
                                                          test_size=0.20,random_state=101)
        clf2=RandomForestClassifier(n_estimators = 150)
        # clf2=svm.SVC(kernel='rbf') 

        clf2.fit(X_train,y_train)
        y_pred=clf2.predict(X_val)
        # print("Ytest",y_test)
        # print("YPred",y_pred)

        print("Accuracy:",accuracy_score(y_test, y_pred))
        
        score = cross_val_score(clf2, X_val, y_test, cv=10)
        print(score.mean())
        clf_report = classification_report(y_test, y_pred)
        # print(clf_report)
        PIK='E:\Fifth Semester\MP\djangoChatbot\chatbot\model\RandomForest.pkl' 
        with open(PIK, "wb") as f:
            pickle.dump(clf2, f)      
        

    def make_prediction(self, test_data, saved_model_name=None):
        try:
            clf = load(str('E:\Fifth Semester\MP\djangoChatbot\chatbot\model\RandomForest.pkl'))
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
        self.symptoms_list= symptoms_dataset.Symptoms.tolist()
        # print("Symptoms",self.symptoms_list,len(self.symptoms_list))
        for symp in self.symptoms_list:
            s=""
            symp=s.join(symp)
            arr=symp.split("_")
            # print(arr,len(arr))
            for i,v in enumerate(arr):
                arr[i]=v.strip()
            final_symps.append(arr)
        # print(final_symps,len(final_symps))
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
        symptoms_dataset.columns=['Symptoms']
        self.symptoms_list= symptoms_dataset.Symptoms.tolist()        
        # print(self.symptoms_list)
        n=len(self.symptoms_list)
        # print(n)
        final_input=[0 for i in range(n)]
        for s in symp:
            i=self.symptoms_list.index(s)
            final_input[i]=1
        print(final_input)
        return final_input



# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]