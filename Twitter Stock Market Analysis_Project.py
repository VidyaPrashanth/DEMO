#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[4]:


df = pd.read_csv('TWTR.csv')
df


# In[5]:


#Data Exploration:


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.info()


# In[9]:


df.isna().sum()


# In[10]:


df['Open'].fillna(df['Open'].mean(), inplace=True)


# In[11]:


df.isna().sum()


# In[12]:


df['High'].fillna(df['High'].mean(), inplace=True)
df['Low'].fillna(df['Low'].mean(), inplace=True)
df['Close'].fillna(df['Close'].mean(), inplace=True)
df['Adj Close'].fillna(df['Adj Close'].mean(), inplace=True)
df['Volume'].fillna(df['Volume'].mean(), inplace=True)


# In[13]:


df.isna().sum()


# In[14]:


df.describe().transpose()


# In[15]:


df_2014 = df[df['Date'].str.startswith('2014')]

# Find the lowest and highest prices in 2014
lowest_price_2014 = df_2014['Low'].min()
highest_price_2014 = df_2014['High'].max()

# Print the lowest and highest prices in 2014
print("Lowest price in 2014: $", lowest_price_2014)
print("Highest price in 2014: $", highest_price_2014)


# In[16]:


df.columns


# In[17]:


#Data Visualization


# In[18]:


# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Plotting the closing price over time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'])
plt.title('Twitter Stock Closing Price (2013-2022)')
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.grid(True)
plt.show()


# In[19]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Open'])
plt.title('Twitter Stock Closing Price (2013-2022)')
plt.xlabel('Year')
plt.ylabel('Opening Price')
plt.grid(True)
plt.show()


# In[20]:


sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Close', data=df)
plt.xticks(rotation=45)
plt.title('Twitter Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Volume', data=df)
plt.xticks(rotation=45)
plt.title('Twitter Stock Volume Traded Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='High', y='Low', data=df)
plt.title('Twitter Stock High vs Low Prices')
plt.xlabel('High Price')
plt.ylabel('Low Price')
plt.show()


# In[21]:


numerical_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Twitter Stock Market Data')
plt.show()


# In[22]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.hist([df['Open'], df['High'], df['Low'], df['Close']], bins=20, stacked=True, label=['Open', 'High', 'Low', 'Close'])
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
ax.set_title('Stacked Histogram of Twitter Stock Prices')
ax.legend()
plt.show()


# In[23]:


sns.set(style="ticks")
g = sns.JointGrid(data=df, x='Date', y='Close', height=6, ratio=5)
g.plot_joint(sns.scatterplot, color='b')
g.plot_marginals(sns.histplot, kde=True, color='b')
plt.xticks(rotation=45, ha='right')
plt.title('Twitter Stock Close Price Scatterplot with Marginal Ticks')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()


# In[ ]:




