#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import warnings


# In[3]:


warnings.filterwarnings("ignore")


# In[4]:


# Load the dataset
house_data = pd.read_csv("D:\House.csv")
house_data


# In[5]:


# Display the first five row
house_data.head()


# In[6]:


# Display the last five row
house_data.tail()


# In[7]:


# Shape of the dataset
house_data.shape


# In[8]:


house_data.shape[0]


# In[9]:


# Column names
house_data.columns


# In[10]:


# Information about the dataset
house_data.info()


# In[11]:


# Statistical description
house_data.describe()


# In[12]:


# Checking for missing values
house_data.isnull().sum()


# In[13]:


# Dropping unnecessary columns
house_data = house_data.drop(['area_type','availability', 'balcony', 'society'], axis = 1)


# In[14]:


house_data.head()


# In[15]:


house_data.isnull().sum()


# In[16]:


# Dropping rows with missing values
house_data = house_data.dropna()
house_data.shape


# In[17]:


# Adding a new column 'BHK'
house_data['BHK'] = house_data['size'].apply(lambda x: int(x.split(' ')[0]))
house_data.head()


# In[18]:


house_data['BHK'].unique()


# In[19]:


house_data['BHK'].value_counts()


# In[20]:


# Plotting correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = house_data.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[21]:


# Plotting pie chart for BHK distribution
import matplotlib.pyplot as plt

sizes = house_data['BHK'].value_counts()
labels = sizes.index

plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Distribution of BHK')
plt.show()


# In[22]:


# Plotting countplot for BHK
plt.figure(figsize=(15,6))
sns.countplot(x='BHK', data = house_data, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[23]:


# Unique values and counts of 'bath'
house_data['bath'].unique()


# In[24]:


house_data['bath'].value_counts()


# In[25]:


# Plotting countplot for bathrooms
plt.figure(figsize=(15,6))
sns.countplot(x='bath', data = house_data, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[26]:


# Investigating rows with more than 15 BHKs
house_data[house_data.BHK > 15].head()


# In[27]:


# Data information
house_data.info()


# In[28]:


# Function to check if a value can be converted to float
def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True


# In[29]:


# Checking non-float values in 'total_sqft'
house_data[house_data['total_sqft'].apply(isfloat)]


# In[30]:


house_data[~house_data['total_sqft'].apply(isfloat)]


# In[31]:


# Function to convert sqft range to a single number
def convert_sqft_num(x):
    token = x.split('-')
    if len(token) == 2:
        return(float(token[0]) + float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[32]:


# Adding 'total_sqft' column
house_data = house_data.copy()
house_data['total_sqft'] = house_data['total_sqft'].apply(convert_sqft_num)
house_data.head()


# In[33]:


#data information
house_data.info()


# In[34]:


house_data.iloc[20]


# In[35]:


# Adding 'price_per_sqft' column
data1 = house_data.copy()
data1['price_per_sqft'] = data1['price']*100000/data1['total_sqft']
data1.head()


# In[36]:


# Cleaning location names and grouping less frequent locations
data1.location.unique()


# In[37]:


data1.location.nunique()


# In[38]:


data1.location = data1.location.apply(lambda x: x.strip())
data1.location


# In[39]:


data1.groupby('location')['location'].count().sort_values(ascending = False)


# In[40]:


location_stats = data1.groupby('location')['location'].count().sort_values(ascending = False)
location_stats


# In[41]:


len(location_stats[location_stats <= 10])


# In[42]:


locationlessthan10 = location_stats[location_stats <= 10]
locationlessthan10


# In[43]:


data1.location.unique()


# In[44]:


data1.location.nunique()


# In[45]:


data1.location = data1.location.apply(lambda x: 'other' if x in locationlessthan10 else x)


# In[46]:


len(data1.location.unique())


# In[47]:


#After cleaning
data1.head()


# In[48]:


# Removing outliers based on sqft per BHK
data1[data1.total_sqft/data1.BHK < 300].head()


# In[49]:


data2 = data1[~(data1.total_sqft/data1.BHK < 300)]
data2.head()


# In[50]:


data2.shape


# In[51]:


data2['price_per_sqft'].describe().apply(lambda x: format(x,'f'))


# In[52]:


# Boxplot for 'price_per_sqft'
plt.figure(figsize=(15,6))
sns.boxplot(x='price_per_sqft', data = data2, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[53]:


# Function to remove outliers based on price per sqft
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft < (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)
    return df_out
data3 = remove_pps_outliers(data2)
data3.shape


# In[54]:


# Scatter plot for 2 BHK and 3 BHK in a location
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.BHK == 2)]
    bhk3 = df[(df.location == location) & (df.BHK == 3)]
    
    plt.rcParams['figure.figsize'] = (8,8)
   
    plt.scatter(bhk2.total_sqft, bhk2.price, color = 'Red', 
                label = '2 BHK', s = 50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color = 'Black', 
                marker = '+', label = '3 BHK', s = 50)
   
    plt.xlabel('Total Square Feet')
    plt.ylabel('Price')
    plt.title(location)
   
    plt.legend()
    plt.show()


# In[55]:


plot_scatter_chart(data3, 'Rajaji Nagar')


# In[56]:


# Boxplot for BHK
plt.figure(figsize=(15,6))
sns.boxplot(x='BHK', data = data2, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[57]:


# Function to remove outliers based on BHK
def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_sats={}
        for BHK,BHK_df in location_df.groupby('BHK'):
            bhk_sats[BHK]={
                'mean':np.mean(BHK_df.price_per_sqft),
                'std':np.std(BHK_df.price_per_sqft),
                'count':BHK_df.shape[0]
            }
        for BHK,BHK_df in location_df.groupby('BHK'):
            stats=bhk_sats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

data4=remove_bhk_outliers(data3)
data4.shape


# In[58]:


plot_scatter_chart(data4, 'Rajaji Nagar')


# In[59]:


# Histogram for price per sqft
plt.rcParams['figure.figsize']=(8,8)
plt.hist(data4.price_per_sqft,rwidth=0.6)
plt.xlabel("Price Per Square Floor")
plt.ylabel("Count")


# In[60]:


data4.bath.unique()


# In[61]:


data4[data4.bath>10]


# In[62]:


plt.rcParams['figure.figsize']=(8,8)
plt.hist(data4.bath,rwidth=0.6)
plt.xlabel("Number Of Bathroom")
plt.ylabel("Count")


# In[63]:


# Removing rows where bathrooms are more than BHK + 2
data4[data4.bath>data4.BHK+2]


# In[64]:


data5=data4[data4.bath<data4.BHK+2]
data5.shape


# In[65]:


data6=data5.drop(['size','price_per_sqft'],axis='columns')
data6


# In[66]:


# Dropping unnecessary columns
data6=data5.drop(['size','price_per_sqft'],axis='columns')
data6


# In[67]:


# Creating dummy variables for 'location'
dummies=pd.get_dummies(data6.location)
dummies.head(10)


# In[68]:


data7=pd.concat([data6,dummies.drop('other',axis='columns')],axis='columns')
data7.head()


# In[69]:


# Dropping 'location' column
data8=data7.drop('location',axis='columns')
data8.head()


# In[70]:


data8.shape


# In[71]:


# Defining features and target variable
X=data8.drop('price',axis='columns')
X.head()


# In[72]:


y=data8.price


# In[74]:


# Splitting the data into training and testing sets
X_train = X.iloc[:5802]
y_train = y.iloc[:5802]
X_test = X.iloc[5802:7252]
y_test = y.iloc[5802:7252]


# In[77]:


# Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

# Define features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set and calculate R^2 score
r2 = model.score(X_test, y_test)
print(f"R^2 score: {r2}")

# Cross-validation using ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv_scores = cross_val_score(model, X, y, cv=cv)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Function to predict price
def price_predict(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]


# In[78]:


price_predict('1st Phase JP Nagar',1000,2,2)


# In[79]:


price_predict('1st Phase JP Nagar',1500,2,3)


# In[80]:


price_predict('5th Phase JP Nagar',1000,2,2)


# In[81]:


#Random Forest Regressor model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Calculate cross-validation score
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Function to predict price
def price_predict(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return rf_model.predict([x])[0]


# In[82]:


price_predict('1st Phase JP Nagar',1000,2,2)


# In[83]:


price_predict('1st Phase JP Nagar',1500,2,3)


# In[84]:


price_predict('5th Phase JP Nagar',1000,2,2)


# In[85]:


# DecisionTreeRegressor model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

# Define features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set and calculate R^2 score
r2 = model.score(X_test, y_test)
print(f"R^2 score: {r2}")

# Cross-validation using ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv_scores = cross_val_score(model, X, y, cv=cv)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Function to predict price
def price_predict(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]


# In[86]:


price_predict('1st Phase JP Nagar',1000,2,2)


# In[87]:


price_predict('1st Phase JP Nagar',1500,2,3)


# In[88]:


price_predict('5th Phase JP Nagar',1000,2,2)


# In[101]:


# XGBRegressor model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

# Define features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBRegressor model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict on the test set and calculate R^2 score
r2 = model.score(X_test, y_test)
print(f"R^2 score: {r2}")

# Cross-validation using ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv_scores = cross_val_score(model, X, y, cv=cv)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Function to predict price
def price_predict(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]


# In[102]:


price_predict('1st Phase JP Nagar',1000,2,2)


# In[103]:


price_predict('1st Phase JP Nagar',1500,2,3)


# In[104]:


price_predict('5th Phase JP Nagar',1000,2,2)


# In[105]:


#Extratree Regression model
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

# Define features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Extra Trees Regressor model
model = ExtraTreesRegressor()
model.fit(X_train, y_train)

# Predict on the test set and calculate R^2 score
r2 = model.score(X_test, y_test)
print(f"R^2 score: {r2}")

# Cross-validation using ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv_scores = cross_val_score(model, X, y, cv=cv)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Function to predict price
def price_predict(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]


# In[106]:


price_predict('1st Phase JP Nagar',1000,2,2)


# In[107]:


price_predict('1st Phase JP Nagar',1500,2,3)


# In[108]:


price_predict('5th Phase JP Nagar',1000,2,2)


# In[109]:


import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assuming the dataset is loaded and preprocessed as before
X = data8.drop('price', axis='columns')
y = data8.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "XGB Regressor": XGBRegressor(),
    "Extra Trees Regressor": ExtraTreesRegressor()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_cv_score = cv_scores.mean()
    results.append({"Model": name, "R² Score": r2, "Mean CV Score": mean_cv_score})

results_df = pd.DataFrame(results)
best_model_name = results_df.loc[results_df["R² Score"].idxmax(), "Model"]
best_model = models[best_model_name]

# Retrain the best model on the entire dataset
best_model.fit(X, y)

# Use the best model to make predictions
def price_predict(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return best_model.predict([x])[0]

price_predict('1st Phase JP Nagar',1000,2,2)


# In[ ]:




