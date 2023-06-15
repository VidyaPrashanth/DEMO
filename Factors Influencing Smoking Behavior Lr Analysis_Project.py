#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.api as sm


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('SmokeBan.csv')
df.head()


# In[5]:


#Descriptive Analysis:


# In[6]:


age_stats = df['age'].describe()
education_counts = df['education'].value_counts()
smoking_counts = df['smoker'].value_counts()

print("Age Summary Statistics:")
print(age_stats)
print("\nEducation Level Counts:")
print(education_counts)
print("\nSmoking Status Counts:")
print(smoking_counts)

smoking_distribution = df['smoker'].value_counts(normalize=True)
print("\nSmoking Distribution:")
print(smoking_distribution)

ban_proportion = df['ban'].value_counts(normalize=True)
print("\nProportion of Individuals Subject to Workplace Smoking Bans:")
print(ban_proportion)


# In[7]:


#Smoking Behavior and Workplace Smoking Bans:


# In[8]:


contingency_table = pd.crosstab(df['smoker'], df['ban'])

chi2, p_value, _, _ = chi2_contingency(contingency_table)

print("Chi-square test results:")
print("Chi-square statistic:", chi2)
print("p-value:", p_value)

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='ban', hue='smoker')
plt.title("Smoking Behavior by Workplace Smoking Bans")
plt.xlabel("Workplace Smoking Bans")
plt.ylabel("Count")
plt.legend(title="Smoker")
plt.show()


# In[9]:


#Demographic Factors and Smoking Behavior:


# In[10]:


smoking_demographics = pd.crosstab([df['smoker']], [df['age'], df['gender'], df['afam'], df['hispanic']])

chi2, p_value, _, _ = chi2_contingency(smoking_demographics)

print("Chi-square test results:")
print("Chi-square statistic:", chi2)
print("p-value:", p_value)

sns.set(style="whitegrid")
plt.figure(figsize=(18, 8))
sns.countplot(data=df, x='age', hue='smoker', palette='coolwarm')
plt.title("Smoking Prevalence by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="Smoker")
plt.show()


# In[11]:


#Education Level and Smoking Behavior:


# In[12]:


smoking_by_education = df.groupby(['education', 'smoker'])['smoker'].count().unstack()

smoking_by_education.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Smoking Prevalence by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Smoker')
plt.show()

smoking_by_education_ban = df.groupby(['education', 'ban', 'smoker'])['smoker'].count().unstack()


smoking_by_education_ban.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Smoking Prevalence by Education Level and Workplace Smoking Bans')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Smoker')
plt.show()

education_smoker = pd.crosstab(df['education'], df['smoker'])
education_ban_smoker = pd.crosstab([df['education'], df['ban']], df['smoker'])

chi2, p_value, _, _ = chi2_contingency(education_smoker)
print('Chi-square test results for education and smoking:')
print(f'Chi-square value: {chi2}')
print(f'p-value: {p_value}')

chi2, p_value, _, _ = chi2_contingency(education_ban_smoker)
print('\nChi-square test results for education, workplace smoking bans, and smoking:')
print(f'Chi-square value: {chi2}')
print(f'p-value: {p_value}')


# In[ ]:




