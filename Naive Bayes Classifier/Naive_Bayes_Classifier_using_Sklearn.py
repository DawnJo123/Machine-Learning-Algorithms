#import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#import data
data = pd.read_csv("Naive Bayes Classifier\dataset\spam.csv")

#Creating a new column called " Spam " to indicate spam as 1 and ham as 0
data['Spam']= data['Category'].apply(lambda x:1 if x=='spam' else 0)

x=data['Message']
y=data['Spam']

# Initializing the CountVectorizer
#CountVectorizer is a text preprocessing technique for converting a collection of text documents into a numerical representation
vectorizer=CountVectorizer()
z=vectorizer.fit_transform(x)

#Spliting data(both x and y) as training data and testing data
X_train,X_test, Y_train, Y_test =train_test_split(z,y, test_size=0.2)

#initializing the Naive Bayes model
nbModel=MultinomialNB()

#Training the model
nbModel.fit(X_train,Y_train)

#Testing the model
prediction=nbModel.predict(X_test)

#Model Evaluation
# Confusion_matrix, accuracy_score, and classification_report are all tools used to evaluate 
# how well a classification model performs by comparing its predictions to the actual labels.
conf_mat=confusion_matrix(Y_test,prediction)
accuracy=accuracy_score(Y_test,prediction)
report=classification_report(Y_test,prediction)

print("Confusion Matrix:\n",conf_mat)
print("\nAccuracy:",accuracy)
print("\n Report:\n",report)
