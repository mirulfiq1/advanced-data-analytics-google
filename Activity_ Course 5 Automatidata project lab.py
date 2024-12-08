#!/usr/bin/env python
# coding: utf-8

# # **Automatidata project**
# **Course 5 - Regression Analysis: Simplify complex data relationships**

# The data consulting firm Automatidata has recently hired you as the newest member of their data analytics team. Their newest client, the NYC Taxi and Limousine Commission (New York City TLC), wants the Automatidata team to build a multiple linear regression model to predict taxi fares using existing data that was collected over the course of a year. The team is getting closer to completing the project, having completed an initial plan of action, initial Python coding work, EDA, and A/B testing.
# 
# The Automatidata team has reviewed the results of the A/B testing. Now it’s time to work on predicting the taxi fare amounts. You’ve impressed your Automatidata colleagues with your hard work and attention to detail. The data team believes that you are ready to build the regression model and update the client New York City TLC about your progress.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # Course 5 End-of-course project: Build a multiple linear regression model
# 
# In this activity, you will build a multiple linear regression model. As you've learned, multiple linear regression helps you estimate the linear relationship between one continuous dependent variable and two or more independent variables. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed. 
# 
# Completing this activity will help you practice planning out and buidling a multiple linear regression model based on a specific business need. The structure of this activity is designed to emulate the proposals you will likely be assigned in your career as a data professional. Completing this activity will help prepare you for those career moments.
# <br/>
# 
# **The purpose** of this project is to demostrate knowledge of EDA and a multiple linear regression model
# 
# **The goal** is to build a multiple linear regression model and evaluate the model
# <br/>
# *This activity has three parts:*
# 
# **Part 1:** EDA & Checking Model Assumptions
# * What are some purposes of EDA before constructing a multiple linear regression model?
# 
# **Part 2:** Model Building and evaluation
# * What resources do you find yourself using as you complete this stage?
# 
# **Part 3:** Interpreting Model Results
# 
# * What key insights emerged from your model(s)?
# 
# * What business recommendations do you propose based on the models built?

# # Build a multiple linear regression model

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## PACE: **Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 

# ### Task 1. Imports and loading
# Import the packages that you've learned are needed for building linear regression models.

# In[115]:


# Imports
# Packages for numerics + dataframes
import numpy as np
import pandas as pd

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for date conversions for calculating trip durations
from datetime import datetime, timedelta

# Packages for OLS (Ordinary Least Squares), MLR (Multiple Linear Regression), confusion matrix
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


# **Note:** `Pandas` is used to load the NYC TLC dataset. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[116]:


# Load dataset into dataframe 
df0=pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv") 


# In[117]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## PACE: **Analyze**
# 
# In this stage, consider the following question where applicable to complete your code response:
# 
# * What are some purposes of EDA before constructing a multiple linear regression model?
# 

# ==> 
# 1. To understand the data first, to know what feature appropriate to use for the analysis that want to do.
# 2. can know the summary of the data.

# ### Task 2a. Explore data with EDA
# 
# Analyze and discover data, looking for correlations, missing data, outliers, and duplicates.

# Start with `.shape` and `.info()`.

# In[118]:


# Start with `.shape` and `.info()`
### YOUR CODE HERE ###
print(df0.shape)
print('---------------------------------------------------------------')
print(df0.info())


# Check for missing data and duplicates using `.isna()` and `.drop_duplicates()`.

# In[119]:


# Check for missing data and duplicates using .isna() and .drop_duplicates()
### YOUR CODE HERE ###
df0.isna().sum()


# In[120]:


df0.drop_duplicates().shape


# Use `.describe()`.

# In[121]:


# Use .describe()
### YOUR CODE HERE ###
df0.describe()


# ### Task 2b. Convert pickup & dropoff columns to datetime
# 

# In[122]:


# Check the format of the data
### YOUR CODE HERE ###
df0['tpep_pickup_datetime'].dtype
df0['tpep_dropoff_datetime'].dtype


# In[123]:


# Convert datetime columns to datetime
### YOUR CODE HERE ###
df0['tpep_pickup_datetime'] = pd.to_datetime(df0['tpep_pickup_datetime'])
df0['tpep_dropoff_datetime'] = pd.to_datetime(df0['tpep_dropoff_datetime'])


# In[124]:


df0['tpep_pickup_datetime'].dtype
df0['tpep_dropoff_datetime'].dtype


# In[125]:


df0


# ### Task 2c. Create duration column

# Create a new column called `duration` that represents the total number of minutes that each taxi ride took.

# In[126]:


# Create `duration` column
### YOUR CODE HERE ###
df0['duration'] = (df0['tpep_dropoff_datetime'] - df0['tpep_pickup_datetime']).dt.total_seconds() / 60


# In[127]:


df0['duration']


# In[128]:


df0


# ### Outliers
# 
# Call `df.info()` to inspect the columns and decide which ones to check for outliers.

# In[129]:


### YOUR CODE HERE ###
df0.info()


# Keeping in mind that many of the features will not be used to fit your model, the most important columns to check for outliers are likely to be:
# * `trip_distance`
# * `fare_amount`
# * `duration`
# 
# 

# ### Task 2d. Box plots
# 
# Plot a box plot for each feature: `trip_distance`, `fare_amount`, `duration`.

# In[130]:


### YOUR CODE HERE ###

# Plot box plots for the columns 'trip_distance', 'fare_amount', and 'duration'
plt.figure(figsize=(15, 5))

# Box plot for 'trip_distance'
plt.subplot(1, 3, 1)
plt.boxplot(df0['trip_distance'].dropna())
plt.title('Trip Distance')
plt.ylabel('Distance')

# Box plot for 'fare_amount'
plt.subplot(1, 3, 2)
plt.boxplot(df0['fare_amount'].dropna())
plt.title('Fare Amount')
plt.ylabel('Amount')

# Box plot for 'duration'
plt.subplot(1, 3, 3)
plt.boxplot(df0['duration'].dropna())
plt.title('Duration')
plt.ylabel('Minutes')

# Show the plots
plt.tight_layout()
plt.show()


# **Questions:** 
# 1. Which variable(s) contains outliers? 
# 
# 2. Are the values in the `trip_distance` column unbelievable?
# 
# 3. What about the lower end? Do distances, fares, and durations of 0 (or negative values) make sense?

# ==> 
# 1. 3 of column contain outliers
# 2. trip distance are more outliers than other
# 3. yes, trip distance have various distance to go to many places. Thats why 0 until 35 km there are must be one people book that kind of distance. The amount of fare seem to have relationship with minutes cause the position of point distribution is likely same of each other
# 4. The zero is totally not make sense for now based on visualization.
# 
# answer:
# 1. All three variables contain outliers. Some are extreme, but others not so much.
# 
# 2. It's 30 miles from the southern tip of Staten Island to the northern end of Manhattan and that's in a straight line. With this knowledge and the distribution of the values in this column, it's reasonable to leave these values alone and not alter them. However, the values for `fare_amount` and `duration` definitely seem to have problematic outliers on the higher end.
# 
# 3. Probably not for the latter two, but for `trip_distance` it might be okay.

# ### Task 2e. Imputations

# #### `trip_distance` outliers
# 
# You know from the summary statistics that there are trip distances of 0. Are these reflective of erroneous data, or are they very short trips that get rounded down?
# 
# To check, sort the column values, eliminate duplicates, and inspect the least 10 values. Are they rounded values or precise values?

# In[131]:


# Are trip distances of 0 bad data or very short trips rounded down?
### YOUR CODE HERE ###
print(df0[['trip_distance', 'fare_amount', 'duration']].head(10))


# In[132]:


df0[df0['trip_distance'] == 0]


# In[133]:


# Are trip distances of 0 bad data or very short trips rounded down?
sorted(set(df0['trip_distance']))[:10]


# In[134]:


sum(df0['trip_distance']==0)


# The distances are captured with a high degree of precision. However, it might be possible for trips to have distances of zero if a passenger summoned a taxi and then changed their mind. Besides, are there enough zero values in the data to pose a problem?
# 
# Calculate the count of rides where the `trip_distance` is zero.

# In[135]:


### YOUR CODE HERE ###
df0[df0['trip_distance'] == 0].count()


# #### `fare_amount` outliers

# In[136]:


df0['fare_amount'].describe()


# In[137]:


### YOUR CODE HERE ###
Q1 = df0['fare_amount'].quantile(0.25)
Q3 = df0['fare_amount'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Determine the outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the outliers
outliers = df0[(df0['fare_amount'] < lower_bound) | (df0['fare_amount'] > upper_bound)]

# Display the outliers
outliers


# **Question:** What do you notice about the values in the `fare_amount` column?
# 
# Impute values less than $0 with `0`.

# In[138]:


# Impute values less than $0 with 0
### YOUR CODE HERE ###
df0[df0['fare_amount'] <= 0]


# In[139]:


# Impute values less than $0 with 0
df0.loc[df0['fare_amount'] < 0, 'fare_amount'] = 0
df0['fare_amount'].min()


# Now impute the maximum value as `Q3 + (6 * IQR)`.

# In[140]:


def outlier_imputer(column_list, iqr_factor):
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.
    '''
    for col in column_list:
        # Reassign minimum to zero
        df0.loc[df0[col] < 0, col] = 0

        # Calculate upper threshold
        q1 = df0[col].quantile(0.25)
        q3 = df0[col].quantile(0.75)
        iqr = q3 - q1
        upper_threshold = q3 + (iqr_factor * iqr)
        print(col)
        print('q3:', q3)
        print('upper_threshold:', upper_threshold)

        # Reassign values > threshold to threshold
        df0.loc[df0[col] > upper_threshold, col] = upper_threshold
        print(df0[col].describe())
        print()


# In[141]:


outlier_imputer(['fare_amount'], 6)


# In[142]:


print(df0.columns)


# In[143]:


print(df0.head())


# In[144]:


df0['duration'].describe()


# #### `duration` outliers
# 

# In[145]:


# Call .describe() for duration outliers
### YOUR CODE HERE ###
### YOUR CODE HERE ###
Q1 = df0['duration'].quantile(0.25)
Q3 = df0['duration'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Determine the outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the outliers
outliers = df0[(df0['duration'] < lower_bound) | (df0['duration'] > upper_bound)]

# Display the outliers
outliers


# In[146]:


df0['duration'].describe()


# The `duration` column has problematic values at both the lower and upper extremities.
# 
# * **Low values:** There should be no values that represent negative time. Impute all negative durations with `0`.
# 
# * **High values:** Impute high values the same way you imputed the high-end outliers for fares: `Q3 + (6 * IQR)`.

# In[147]:


# Impute a 0 for any negative values
### YOUR CODE HERE ###
df0[df0['duration'] <= 0]


# In[148]:


df0.loc[df0['duration'] < 0, 'duration'] = 0
df0['duration'].min()


# In[149]:


# Impute the high outliers
### YOUR CODE HERE ###
outlier_imputer(['duration'], 6)


# ### Task 3a. Feature engineering

# #### Create `mean_distance` column
# 
# When deployed, the model will not know the duration of a trip until after the trip occurs, so you cannot train a model that uses this feature. However, you can use the statistics of trips you *do* know to generalize about ones you do not know.
# 
# In this step, create a column called `mean_distance` that captures the mean distance for each group of trips that share pickup and dropoff points.
# 
# For example, if your data were:
# 
# |Trip|Start|End|Distance|
# |--: |:---:|:-:|    |
# | 1  | A   | B | 1  |
# | 2  | C   | D | 2  |
# | 3  | A   | B |1.5 |
# | 4  | D   | C | 3  |
# 
# The results should be:
# ```
# A -> B: 1.25 miles
# C -> D: 2 miles
# D -> C: 3 miles
# ```
# 
# Notice that C -> D is not the same as D -> C. All trips that share a unique pair of start and end points get grouped and averaged.
# 
# Then, a new column `mean_distance` will be added where the value at each row is the average for all trips with those pickup and dropoff locations:
# 
# |Trip|Start|End|Distance|mean_distance|
# |--: |:---:|:-:|  :--   |:--   |
# | 1  | A   | B | 1      | 1.25 |
# | 2  | C   | D | 2      | 2    |
# | 3  | A   | B |1.5     | 1.25 |
# | 4  | D   | C | 3      | 3    |
# 
# 
# Begin by creating a helper column called `pickup_dropoff`, which contains the unique combination of pickup and dropoff location IDs for each row.
# 
# One way to do this is to convert the pickup and dropoff location IDs to strings and join them, separated by a space. The space is to ensure that, for example, a trip with pickup/dropoff points of 12 & 151 gets encoded differently than a trip with points 121 & 51.
# 
# So, the new column would look like this:
# 
# |Trip|Start|End|pickup_dropoff|
# |--: |:---:|:-:|  :--         |
# | 1  | A   | B | 'A B'        |
# | 2  | C   | D | 'C D'        |
# | 3  | A   | B | 'A B'        |
# | 4  | D   | C | 'D C'        |
# 

# In[150]:


# Create `pickup_dropoff` column
### YOUR CODE HERE ###
df0['pickup_dropoff'] = df0['PULocationID'].astype(str) + ' ' + df0['DOLocationID'].astype(str)
df0['pickup_dropoff'].head(2)


# Now, use a `groupby()` statement to group each row by the new `pickup_dropoff` column, compute the mean, and capture the values only in the `trip_distance` column. Assign the results to a variable named `grouped`.

# In[151]:


### YOUR CODE HERE ###
grouped = df0.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
grouped[:5]


# `grouped` is an object of the `DataFrame` class.
# 
# 1. Convert it to a dictionary using the [`to_dict()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html) method. Assign the results to a variable called `grouped_dict`. This will result in a dictionary with a key of `trip_distance` whose values are another dictionary. The inner dictionary's keys are pickup/dropoff points and its values are mean distances. This is the information you want.
# 
# ```
# Example:
# grouped_dict = {'trip_distance': {'A B': 1.25, 'C D': 2, 'D C': 3}
# ```
# 
# 2. Reassign the `grouped_dict` dictionary so it contains only the inner dictionary. In other words, get rid of `trip_distance` as a key, so:
# 
# ```
# Example:
# grouped_dict = {'A B': 1.25, 'C D': 2, 'D C': 3}
#  ```

# In[152]:


# 1. Convert `grouped` to a dictionary
grouped_dict = grouped.to_dict()

# 2. Reassign to only contain the inner dictionary
grouped_dict = grouped_dict['trip_distance']


# 1. Create a `mean_distance` column that is a copy of the `pickup_dropoff` helper column.
# 
# 2. Use the [`map()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html#pandas-series-map) method on the `mean_distance` series. Pass `grouped_dict` as its argument. Reassign the result back to the `mean_distance` series.
# </br></br>
# When you pass a dictionary to the `Series.map()` method, it will replace the data in the series where that data matches the dictionary's keys. The values that get imputed are the values of the dictionary.
# 
# ```
# Example:
# df['mean_distance']
# ```
# 
# |mean_distance |
# |  :-:         |
# | 'A B'        |
# | 'C D'        |
# | 'A B'        |
# | 'D C'        |
# | 'E F'        |
# 
# ```
# grouped_dict = {'A B': 1.25, 'C D': 2, 'D C': 3}
# df['mean_distance`] = df['mean_distance'].map(grouped_dict)
# df['mean_distance']
# ```
# 
# |mean_distance |
# |  :-:         |
# | 1.25         |
# | 2            |
# | 1.25         |
# | 3            |
# | NaN          |
# 
# When used this way, the `map()` `Series` method is very similar to `replace()`, however, note that `map()` will impute `NaN` for any values in the series that do not have a corresponding key in the mapping dictionary, so be careful.

# In[153]:


# 1. Create a mean_distance column that is a copy of the pickup_dropoff helper column
df0['mean_distance'] = df0['pickup_dropoff']

# 2. Map `grouped_dict` to the `mean_distance` column
df0['mean_distance'] = df0['mean_distance'].map(grouped_dict)

# Confirm that it worked
df0[(df0['PULocationID']==100) & (df0['DOLocationID']==231)][['mean_distance']]


# #### Create `mean_duration` column
# 
# Repeat the process used to create the `mean_distance` column to create a `mean_duration` column.

# In[154]:


grouped = df0.groupby('pickup_dropoff').mean(numeric_only=True)[['duration']]
grouped

# Create a dictionary where keys are unique pickup_dropoffs and values are
# mean trip duration for all trips with those pickup_dropoff combos
grouped_dict = grouped.to_dict()
grouped_dict = grouped_dict['duration']

df0['mean_duration'] = df0['pickup_dropoff']
df0['mean_duration'] = df0['mean_duration'].map(grouped_dict)

# Confirm that it worked
df0[(df0['PULocationID']==100) & (df0['DOLocationID']==231)][['mean_duration']]


# #### Create `day` and `month` columns
# 
# Create two new columns, `day` (name of day) and `month` (name of month) by extracting the relevant information from the `tpep_pickup_datetime` column.

# In[155]:


df = df0.copy()


# In[156]:


# Create 'day' col
df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()

# Create 'month' col
df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()


# #### Create `rush_hour` column
# 
# Define rush hour as:
# * Any weekday (not Saturday or Sunday) AND
# * Either from 06:00&ndash;10:00 or from 16:00&ndash;20:00
# 
# Create a binary `rush_hour` column that contains a 1 if the ride was during rush hour and a 0 if it was not.

# In[157]:


# Create 'rush_hour' col
df['rush_hour'] = df['tpep_pickup_datetime'].dt.hour

# If day is Saturday or Sunday, impute 0 in `rush_hour` column
df.loc[df['day'].isin(['saturday', 'sunday']), 'rush_hour'] = 0


# In[158]:


def rush_hourizer(hour):
    if 6 <= hour['rush_hour'] < 10:
        val = 1
    elif 16 <= hour['rush_hour'] < 20:
        val = 1
    else:
        val = 0
    return val


# In[159]:


# Apply the `rush_hourizer()` function to the new column
df.loc[(df.day != 'saturday') & (df.day != 'sunday'), 'rush_hour'] = df.apply(rush_hourizer, axis=1)
df.head()


# ### Task 4. Scatter plot
# 
# Create a scatterplot to visualize the relationship between `mean_duration` and `fare_amount`.

# In[160]:


# Create a scatter plot of duration and trip_distance, with a line of best fit
sns.set(style='whitegrid')
f = plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
sns.regplot(x=df['mean_duration'], y=df['fare_amount'],
            scatter_kws={'alpha':0.5, 's':5},
            line_kws={'color':'red'})
plt.ylim(0, 70)
plt.xlim(0, 70)
plt.title('Mean duration x fare amount')
plt.show()


# The `mean_duration` variable correlates with the target variable. But what are the horizontal lines around fare amounts of 52 dollars and 63 dollars? What are the values and how many are there?
# 
# You know what one of the lines represents. 62 dollars and 50 cents is the maximum that was imputed for outliers, so all former outliers will now have fare amounts of \$62.50. What is the other line?
# 
# Check the value of the rides in the second horizontal line in the scatter plot.

# In[161]:


df[df['fare_amount'] > 50]['fare_amount'].value_counts().head()


# Examine the first 30 of these trips.

# In[162]:


# Set pandas to display all columns
pd.set_option('display.max_columns', None)
df[df['fare_amount']==52].head(30)


# **Question:** What do you notice about the first 30 trips?
# 
# ==> ENTER YOUR RESPONSE HERE

# ### Task 5. Isolate modeling variables
# 
# Drop features that are redundant, irrelevant, or that will not be available in a deployed environment.

# In[163]:


df.info()


# In[164]:


df2 = df.copy()

df2 = df2.drop(['Unnamed: 0', 'tpep_dropoff_datetime', 'tpep_pickup_datetime',
               'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
               'payment_type', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
               'total_amount', 'tpep_dropoff_datetime', 'tpep_pickup_datetime', 'duration',
               'pickup_dropoff', 'day', 'month'
               ], axis=1)

df2.info()


# ### Task 6. Pair plot
# 
# Create a pairplot to visualize pairwise relationships between `fare_amount`, `mean_duration`, and `mean_distance`.

# In[ ]:


# Create a pairplot to visualize pairwise relationships between variables in the data
### YOUR CODE HERE ###

sns.pairplot(df2[['fare_amount', 'mean_duration', 'mean_distance']],
             plot_kws={'alpha':0.4, 'size':5},
             );


# These variables all show linear correlation with each other. Investigate this further.

# ### Task 7. Identify correlations

# Next, code a correlation matrix to help determine most correlated variables.

# In[ ]:


# Create correlation matrix containing pairwise correlation of columns, using pearson correlation coefficient
df2.corr(method='pearson')


# Visualize a correlation heatmap of the data.

# In[ ]:


# Create correlation heatmap

plt.figure(figsize=(6,4))
sns.heatmap(df2.corr(method='pearson'), annot=True, cmap='Reds')
plt.title('Correlation heatmap',
          fontsize=18)
plt.show()


# **Question:** Which variable(s) are correlated with the target variable of `fare_amount`? 
# 
# Try modeling with both variables even though they are correlated.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## PACE: **Construct**
# 
# After analysis and deriving variables with close relationships, it is time to begin constructing the model. Consider the questions in your PACE Strategy Document to reflect on the Construct stage.
# 

# ### Task 8a. Split data into outcome variable and features

# In[ ]:


### YOUR CODE HERE ###

df2.info()


# Set your X and y variables. X represents the features and y represents the outcome (target) variable.

# In[ ]:


# Remove the target column from the features
X = df2.drop(columns=['fare_amount'])

# Set y variable
y = df2[['fare_amount']]

# Display first few rows
X.head()


# ### Task 8b. Pre-process data
# 

# Dummy encode categorical variables

# In[ ]:


# Convert VendorID to string
X['VendorID'] = X['VendorID'].astype(str)

# Get dummies
X = pd.get_dummies(X, drop_first=True)
X.head()


# ### Split data into training and test sets

# Create training and testing sets. The test set should contain 20% of the total samples. Set `random_state=0`.

# In[ ]:


# Create training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ### Standardize the data
# 
# Use `StandardScaler()`, `fit()`, and `transform()` to standardize the `X_train` variables. Assign the results to a variable called `X_train_scaled`.

# In[ ]:


# Standardize the X variables
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train scaled:', X_train_scaled)


# ### Fit the model
# 
# Instantiate your model and fit it to the training data.

# In[ ]:


# Fit your model to the training data
lr=LinearRegression()
lr.fit(X_train_scaled, y_train)


# ### Task 8c. Evaluate model

# ### Train data
# 
# Evaluate your model performance by calculating the residual sum of squares and the explained variance score (R^2). Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# In[ ]:


# Evaluate the model performance on the training data
r_sq = lr.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)
y_pred_train = lr.predict(X_train_scaled)
print('R^2:', r2_score(y_train, y_pred_train))
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print('MSE:', mean_squared_error(y_train, y_pred_train))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred_train)))


# ### Test data
# 
# Calculate the same metrics on the test data. Remember to scale the `X_test` data using the scaler that was fit to the training data. Do not refit the scaler to the testing data, just transform it. Call the results `X_test_scaled`.

# In[ ]:


# Scale the X_test data
X_test_scaled = scaler.transform(X_test)


# In[ ]:


# Evaluate the model performance on the testing data
r_sq_test = lr.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq_test)
y_pred_test = lr.predict(X_test_scaled)
print('R^2:', r2_score(y_test, y_pred_test))
print('MAE:', mean_absolute_error(y_test,y_pred_test))
print('MSE:', mean_squared_error(y_test, y_pred_test))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred_test)))


# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## PACE: **Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### Task 9a. Results
# 
# Use the code cell below to get `actual`,`predicted`, and `residual` for the testing set, and store them as columns in a `results` dataframe.

# In[ ]:


# Create a `results` dataframe
results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                             'predicted': y_pred_test.ravel()})
results['residual'] = results['actual'] - results['predicted']
results.head()


# ### Task 9b. Visualize model results

# Create a scatterplot to visualize `actual` vs. `predicted`.

# In[ ]:


# Create a scatterplot to visualize `predicted` over `actual`
fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x='actual',
                y='predicted',
                data=results,
                s=20,
                alpha=0.5,
                ax=ax
)
# Draw an x=y line to show what the results would be if the model were perfect
plt.plot([0,60], [0,60], c='red', linewidth=2)
plt.title('Actual vs. predicted');


# Visualize the distribution of the `residuals` using a histogram.

# In[ ]:


# Visualize the distribution of the `residuals`
sns.histplot(results['residual'], bins=np.arange(-15,15.5,0.5))
plt.title('Distribution of the residuals')
plt.xlabel('residual value')
plt.ylabel('count');


# In[ ]:


# Calculate residual mean
### YOUR CODE HERE ###
results['residual'].mean()


# Create a scatterplot of `residuals` over `predicted`.

# In[ ]:


# Create a scatterplot of `residuals` over `predicted`
### YOUR CODE HERE ###
# Create a scatterplot of `residuals` over `predicted`

sns.scatterplot(x='predicted', y='residual', data=results)
plt.axhline(0, c='red')
plt.title('Scatterplot of residuals over predicted values')
plt.xlabel('predicted value')
plt.ylabel('residual value')
plt.show()


# ### Task 9c. Coefficients
# 
# Use the `coef_` attribute to get the model's coefficients. The coefficients are output in the order of the features that were used to train the model. Which feature had the greatest effect on trip fare?

# In[ ]:


# Get model coefficients
coefficients = pd.DataFrame(lr.coef_, columns=X.columns)
coefficients


# What do these coefficients mean? How should they be interpreted?

# ==> 
# The coefficients reveal that `mean_distance` was the feature with the greatest weight in the model's final prediction. Be careful here! A common misinterpretation is that for every mile traveled, the fare amount increases by a mean of \\$7.13. This is incorrect. Remember, the data used to train the model was standardized with `StandardScaler()`. As such, the units are no longer miles. In other words, you cannot say "for every mile traveled...", as stated above. The correct interpretation of this coefficient is: controlling for other variables, *for every +1 change in standard deviation*, the fare amount increases by a mean of \\$7.13. 
# 
# Note also that because some highly correlated features were not removed, the confidence interval of this assessment is wider.
# 
# So, translate this back to miles instead of standard deviation (i.e., unscale the data). 
# 
# 1. Calculate the standard deviation of `mean_distance` in the `X_train` data.
# 
# 2. Divide the coefficient (7.133867) by the result to yield a more intuitive interpretation.

# In[ ]:


# 1. Calculate SD of `mean_distance` in X_train data
print(X_train['mean_distance'].std())

# 2. Divide the model coefficient by the standard deviation
print(7.133867 / X_train['mean_distance'].std())


# ### Task 9d. Conclusion
# 
# 1. What are the key takeaways from this notebook?
# 
# * Multiple linear regression is a powerful tool to estimate a dependent continous variable from several independent variables.
# * Exploratory data analysis is useful for selecting both numeric and categorical features for multiple linear regression.
# * Fitting multiple linear regression models may require trial and error to select variables that fit an accurate model while maintaining model assumptions (or not, depending on your use case).
# 
# 2. What results can be presented from this notebook?
# 
# *  You can discuss meeting linear regression assumptions, and you can present the MAE and RMSE scores obtained from the model.
# 
# 

# ==> ENTER YOUR RESPONSE HERE 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged. 
