# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries. 

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively. 

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset. 


5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
~~~

## Developed by: Manoj M
## RegisterNumber: 212221240027

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


~~~

## Output:

### Placement Data:

![1](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/9a71b246-d3a8-4e95-ac78-a9548652743e)

### Salary Data:

![2](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/0a329819-7e64-463a-b34f-4816537418a4)

### Checking the null() function:

![3](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/10357f08-fe00-4c95-97b2-999e2dd62faa)

### Data Duplicate:

![4](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/20a9e9d0-6b72-4349-a128-27b7484b1e9f)

### Print Data:

![5](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/3ee9155f-19f3-4107-80aa-2246bfc2c025)

### Data-Status:

![6](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/c9271330-f373-4151-9d9c-2e9f95380d77)

### Y_prediction array:

![7](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/da111b16-5f5e-4e27-a647-cddd7e927498)

### Accuracy value:

![8](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/8244b5de-96bf-4f66-b3ab-a368193df414)

### Confusion array:

![9](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/48f539d1-9809-4524-a2f9-13428593d70d)


### Classification Report:

![10](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/21269674-a58b-46b0-95fd-73fe55a8c234)



### Prediction of LR:

![11](https://github.com/Manoj21500566/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94588708/2dee8048-aa2e-45c8-9bb0-bb2dd1d2cf35)







## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
