
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df1=pd.read_csv("./dataset/dataset.csv")
df1.shape


from sklearn.feature_extraction.text import CountVectorizer

df1 = df1.replace(np.nan, '', regex=True)
df3=df1.drop(['Disease'],axis='columns')
df3 = df1['Symptom_1'].map(str) + ' ' + df1['Symptom_2'].map(str) + ' ' + df1['Symptom_3'].map(str)+ ' ' + df1['Symptom_4'].map(str)+ ' ' + df1['Symptom_5'].map(str)+ ' ' + df1['Symptom_6'].map(str)+ ' ' + df1['Symptom_7'].map(str)+ ' ' + df1['Symptom_8'].map(str)+ ' ' + df1['Symptom_9'].map(str)+ ' ' + df1['Symptom_10'].map(str)+ ' ' + df1['Symptom_11'].map(str)+ ' ' + df1['Symptom_12'].map(str)+ ' ' + df1['Symptom_13'].map(str)+ ' ' + df1['Symptom_14'].map(str)+ ' ' + df1['Symptom_15'].map(str)+ ' ' + df1['Symptom_16'].map(str)+ ' ' + df1['Symptom_17'].map(str)
corpus=df3.tolist()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
headings=vectorizer.get_feature_names()
symptoms_list=headings
headings.append('prognosis')

X=X.toarray()

df4 = pd.DataFrame(X)
df5=pd.concat([df4,df1.Disease],axis='columns')
df5.columns=headings

df5.to_csv('./dataset/disease_symptom_mapping.csv',index=False)
pd.DataFrame(symptoms_list).to_csv("./dataset/Symptoms.csv")
