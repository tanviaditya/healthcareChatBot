from main import DiseasePrediction
import random
from googletrans import Translator
import csv
import pandas as pd

def predictDisease(symp):
    dp = DiseasePrediction()
    dp.train_model()
    test=dp.inputNLP(symp)
    test_data=[test]
    result=dp.make_prediction(test_data,saved_model_name=current_model_name)
    return result

def getDescription(disease):
    dscp=pd.read_csv("./dataset/symptom_Description.csv")
    r=dscp.loc[dscp['Disease']==disease]
    return r
def getPrecautions(disease):
    dscp=pd.read_csv("./dataset/symptom_precaution.csv")
    r=dscp.loc[dscp['Disease']==disease]
    return r

    

if __name__ == "__main__":
    translator = Translator()
    greetings=["Hi","hi","Hi there", "How are you ?", "Is anyone there?","Hey","Hola", "Hello", "Good day","Hey"]
    greetings_response=["Hello", "Good to see you again", "Hi there, How can I help?","Hey"]
    exit_list=["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time","Thanks","Thank you"]
    exit_respose= ["See you!", "Have a nice day", "Bye!","Bye"]
    symptoms=list()
    flag=0
    print("Type 1 for English")
    print("हिंदी के लिए 2 दबाएं")
    print("ગુજરાતી માટે 3 દબાવો")
    l=int(input())
    lang=""
    if l==1:
        lang='en'
    elif l==2:
        lang='hi'
    elif l==3:
        lang='gu'

    user="User: "
    hcb="Health Bot: "
    if lang!='en':
        user=translator.translate(user, src='en', dest=lang).text
        hcb=translator.translate(hcb, src='en', dest=lang).text
    while(True):
        i=input(user)
        if lang!='en':
            i=translator.translate(i, src=lang, dest='en').text
        if i in exit_list:
            result = random.choice(exit_respose)
            result=translator.translate(result, src='en', dest=lang).text
            print(hcb+result)
            break
        elif i in greetings:
            result = random.choice(greetings_response)
            result=translator.translate(result, src='en', dest=lang).text
            print(hcb+result)
        else:
            if i=="":
                s="HealthBot: Please enter something.."
                if lang!='en':
                    s=translator.translate(s, src='en', dest=lang).text
                print(s)
            else:
                current_model_name = 'mnb'
                dp = DiseasePrediction()
                arr=dp.symptomDetector(i)
                symptoms=symptoms+arr
                s_list=set(symptoms)
                symptoms=list(s_list)
                if len(symptoms)==0:
                    s="Please Enter symptoms you are experiencing"
                    s=translator.translate(s, src='en', dest=lang).text
                    print(hcb+s)
                elif len(symptoms)<3 and flag<2:
                    s="Are there any other symptoms I should know about? "
                    s=translator.translate(s, src='en', dest=lang).text
                    print(hcb+s)
                    flag+=1
                else:
                    res=predictDisease(symptoms)
                    res1=res[0]
                    s="You maybe suffering from "
                    if lang!='en':
                        s=translator.translate(s, src='en', dest=lang).text
                        res1=translator.translate(res1, src='en', dest=lang).text
                    print("Symptoms: ")
                    print(symptoms)
                    print(hcb+s+" "+res1)
                    d=getDescription(res[0])
                    d=d['Description'].values[0]
                    if lang!='en':
                        d=translator.translate(d, src='en', dest=lang).text
                    des="Description: "
                    if lang!='en':
                        des=translator.translate(des, src='en', dest=lang).text
                    print(des+d)

                    r=getPrecautions(res[0])
                    r1=r['Precaution_1'].values[0].title()
                    r2=r['Precaution_2'].values[0].title()
                    r3=r['Precaution_3'].values[0].title()
                    r4=r['Precaution_4'].values[0].title()
                    if lang!='en':
                        r1=translator.translate(r1, src='en', dest=lang).text
                        r2=translator.translate(r2, src='en', dest=lang).text
                        r3=translator.translate(r3, src='en', dest=lang).text
                        r4=translator.translate(r4, src='en', dest=lang).text
                    
                    pre="Precautions: "
                    if lang!='en':
                        pre=translator.translate(pre, src='en', dest=lang).text
                    print(pre)
                    print("1] "+r1)
                    print("2] "+r2)
                    print("3] "+r3)
                    print("4] "+r4)





# નમસ્તે
# હું બીમાર છું
# હું શરદી અને માથાનો દુખાવો અનુભવી રહ્યો છું
# મને ઉબકા, ભૂખ ઓછી થવી, સાંધાનો દુખાવો, ત્વચા પર ફોલ્લીઓ, ઉલટી અને વધુ તાવ છે
# આભાર