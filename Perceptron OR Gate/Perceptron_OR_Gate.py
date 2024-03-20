#Import libraries
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

#Input and Output data for OR gate.
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,1,1,1]     # Output corresponding to each input pair

#Creating the model
ppn= Perceptron()

#Training the nodel using input and output data.
ppn.fit(X,Y)

#Predicting the outputs for the input data.
predicted_output=ppn.predict(X)

#Calculate the accuracy by comparing predicted outputs with actual outputs
accuracy= accuracy_score(Y, predicted_output)

print("Predicted Outputs: ", predicted_output)
print("Accuracy: ", accuracy)

#Entering a new input and checking model efficiency
new_input=[[1,0]]
predicted_output_new=ppn.predict(new_input)
print(f"Predicted output for {new_input}: {predicted_output_new}")