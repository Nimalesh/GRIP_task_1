#!/usr/bin/env python
# coding: utf-8

# Name : Nimalesh E
# Data Science and Business Analytics Intern
# Task 1 : Prediction using Supervised ML
# Problem : Predict the score of a student based on the number of study hours.

# In[13]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(10)


# In[15]:


data.describe()


# In[16]:


data.shape


# In[17]:


correlation=data.corr(method='pearson')
print(correlation)


# In[18]:



data.plot(x='Hours',y='Scores',style='*')
plt.title('Scatter plot of Hours Vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[19]:


#Assigning the attributes to x and labels to y
X=data.iloc[:,:-1].values
y=data.iloc[:,1].values

#Splitting the dataset for training and testing of model
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

#Training the algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

#Plotting the regression line
reg=regressor.coef_*X+regressor.intercept_

#Plotting for the test data
plt.figure(figsize=(9,7))
plt.scatter(X,y)
plt.plot(X,reg);
plt.title("Regression Line")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()


# In[20]:


#Using model to make predictions
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual Scores': y_test, 'Predicted Scores': y_pred})  
df


# In[21]:



hour=9.25 #Provided Value
h=np.array(hour) #convert to a numpy array
h=h.reshape(1,1)
prediction=regressor.predict(h)
print("The number of hours studied is = {}".format(hour))
print("The predicted score is = {}".format(prediction[0]))


# In[22]:


#Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
mse=(mean_absolute_error(y_test,y_pred))
print("The Mean Absolute Error is =",mse)
rmse=(np.sqrt(mean_squared_error(y_test,y_pred)))
print("The Root Mean Square Error is =",rmse)

#To find coefficient of determination
r2=r2_score(y_test,y_pred)
print("The R-square is =",r2)


# In[ ]:





# In[ ]:




