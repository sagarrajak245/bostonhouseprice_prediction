# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# %%




# Load the dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'price']
budget_df = pd.read_csv('housing.csv', header=None, names=column_names, delim_whitespace=True)

# Display the first few rows of the dataset
print(budget_df.head())





# %%
print(budget_df.tail())




# %%
# Display basic information about the dataset
print(budget_df.info())


# %%
budget_stats=budget_df.describe()
print(budget_stats)


# %%
#check is null dataset
print(budget_df.isnull().sum())






# %%
# Count number  duplicate rows 
print(budget_df.duplicated().sum())










# %%
#correlation of dataset
corr_data= budget_df.corr()
print(corr_data)

# %%

#check multi collinearity 
plt.figure(figsize=(12, 8))
sns.heatmap(budget_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()






# %%
#extract target variable

budget_target=budget_df['price']
print(  budget_target)

# %%
#extract target variable
budget_df= budget_df.drop('price',axis=1)
print(budget_df)
# print(budget_df.columns)













# %%
#pairplot  each data with each other
sns.pairplot(budget_df)

# %%
#plot of price vs main features like CRIM,RM,LSTAT,PTRATIO
 # Plot of price vs CRIM
sns.regplot(x=budget_df['CRIM'],y=budget_target,color='aqua')
plt.title('Price vs CRIM')
plt.xlabel('CRIM')
plt.ylabel('Price')
plt.show()

sns.scatterplot(x=budget_df['CRIM'],y=budget_target,color='red')








# %%
#plot of price vs RM
sns.regplot(x=budget_df['RM'],y=budget_target)
plt.title('Price vs RM')
plt.xlabel('RM')
plt.ylabel('Price')
plt.show()



sns.scatterplot(x=budget_df['RM'],y=budget_target,color='red')

# %%
#plot of price vs LSTAT
sns.regplot(x=budget_df['LSTAT'],y=budget_target)
plt.title('Price vs LSTAT')
plt.xlabel('LSTAT')
plt.ylabel('Price')
plt.show()




sns.scatterplot(x=budget_df['LSTAT'],y=budget_target,color='red')

# %%
#plot price vs PTRATIO
sns.regplot(x=budget_df['PTRATIO'],y=budget_target)
plt.title('Price vs PTRATIO')
plt.xlabel('PTRATIO')
plt.ylabel('Price')
plt.show()





sns.scatterplot(x=budget_df['PTRATIO'],y=budget_target,color='red')

# %%
# iqr
q1=budget_stats.iloc[4]
q3=budget_stats.iloc[6]
iqr=q3-q1
print(iqr)











# %%
#lower bound
lower_bound=q1-1.5*iqr
#upper bound
upper_bound=q3+1.5*iqr
print("Lower_bound:",lower_bound,"\nUpper_bound:",upper_bound)






# %%
#step1 complete preprocessing
#step2 split the dataset into training and testing
#step3 train the model
#step4 test the model
#step5 evaluate the model

#step6 predict the model
#step7 save the model
#step8 load the model
#step9 use the model to predict the price
#step10 plot the predicted price vs actual price









# %%
#step2 split the dataset into training and testing
x=budget_df
y=budget_target

print("x:",x)
print("y:",y)




















# %%
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression



#split the dataset into training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



#stanadarization
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


#linear regression model
#apply lasso,ridge,elasticnet if you want to apply regularization more detail here no use 
#already applied ridge no use

model=LinearRegression()

#cv score  it folds set into group and applies cross validation
cv_score=cross_val_score(model,x_train,y_train,cv=7,scoring='neg_mean_squared_error')
cv_score=-cv_score
print("cv_score:",cv_score)
print("mean cv_score:",cv_score.mean())





















# %%
# now fit model  ytrain is not fitted bcoz we want to eavluateon that data
model.fit(X_train_scaled,y_train)

#predict the model
y_pred=model.predict(X_test_scaled)


#calculate metrics
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,r2_score

mse=mean_squared_error(y_test,y_pred)
print("mse:",mse)

rmse=np.sqrt(mse)
print("rmse:",rmse)

mae=mean_absolute_error(y_test,y_pred)
print("mae:",mae)

r2=r2_score(y_test,y_pred)
print("r2:",r2)


print("adjuster r2:",1-(1-r2)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))





# For R-squared and Adjusted R-squared, higher is better (closer to 1).
# For MSE, RMSE, MAE, and MAPE, lower is better (closer to 0).






# %%
#above 0.5 r2 is good score 
#above 0.75 r2 is very good score
#above 0.90 r2 is excellent score
#adjusted r2 is always less than r2

#residue

residual=y_test-y_pred
print("residual:",residual)

#plot of residual
plt.figure(figsize=(10,6))
plt.scatter(y_pred,residual)
plt.axhline(y=0,color='red',linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.show()

#normal distribution
sns.displot(residual,kde=True)
plt.title('Normal Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

#it is normally distributed
#it is used to check the accuracy of the model


# %%
# 2. Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted')
plt.show()

# %%
#prediction on new data
new_data=budget_df.iloc[:1].values.reshape(1,-1)
print("new_data:",new_data)


#scale new data
new_data_scaled=scaler.transform(new_data)
print("new_data_scaled:",new_data_scaled)
#fit the model
model.fit(X_train_scaled,y_train)


#predict the price
predicted_price=model.predict(new_data_scaled)
print("predicted_price:",predicted_price[0])






# %%
#save model
import joblib
joblib.dump(model,'boston_house_price_prediction.pkl')
print("\nModel saved as 'boston_housing_model.pkl'")


#load model
job_model=joblib.load('boston_house_price_prediction.pkl')
print("\nModel loaded from 'boston_housing_model.pkl'")

#predict the price
new_data=budget_df.iloc[:1].values.reshape(1,-1)
print("new_data:",new_data)

#scale new data
new_data_scaled=scaler.transform(new_data)
print("new_data_scaled:",new_data_scaled)


#predict the price
predicted_price=job_model.predict(new_data_scaled)
print("predicted_price:",predicted_price[0])




#it is working great






