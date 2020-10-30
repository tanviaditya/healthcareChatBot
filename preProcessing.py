
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df1=pd.read_csv("./dataset/dataset.csv")
df1.shape


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
# df5.drop_duplicates(keep='first',inplace=True)
df5.to_csv('./dataset/disease_symptom_mapping.csv',index=False)

# #feature selection
# X, y = df5.iloc[:,:-1], df5.iloc[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
# clf=RandomForestClassifier(n_estimators=100)
# clf.fit(X_train,y_train)
# feature_imp = pd.Series(clf.feature_importances_,index=list(df5.columns[:-1])).sort_values(ascending=False).head(55)
# index=feature_imp[::-1].index
# index=index.tolist()
# X_reduced, y = df5[index], df5.iloc[:,-1]

# final=pd.concat([X_reduced,y],axis='columns')
# final.to_csv('./dataset/final.csv',index=False)

symp=pd.DataFrame(symptoms_list)
symp.columns=['Symptoms']
last_row = len(symp)-1
symp = symp.drop(symp.index[last_row]) 
symp.to_csv('./dataset/Symptoms.csv',index=False)


# pd.DataFrame(symptoms_list).to_csv("./dataset/Symptoms.csv")