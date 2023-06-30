#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

warnings.filterwarnings('ignore')


# In[ ]:


#2.Load Dataset


# In[2]:


df = pd.read_csv("breast-cancer.csv")
df.drop('id', axis=1, inplace=True)
df


# In[ ]:


#3. Simple EDA


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[ ]:


#Visualization


# In[6]:


df['diagnosis'].value_counts().plot(kind='bar', color=['blue', 'red'])
print(df['diagnosis'].value_counts())
plt.show()


# In[7]:


df_encoded = df.copy()

label_encoder = LabelEncoder()
df_encoded['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

correlation = df_encoded.corr()

plt.figure(figsize=(26, 20))
sns.heatmap(correlation, annot=True, cmap="winter")
plt.show()


# In[8]:


corr_features = df_encoded.corr()['diagnosis'] > 0.70
df_subset = df_encoded[corr_features.index[corr_features]]
df_subset['diagnosis'] = df_encoded['diagnosis']

g = sns.pairplot(df_subset, diag_kind='kde', markers='+', hue='diagnosis', palette=('r', 'b'),
             plot_kws=dict(s=25, edgecolor='b', linewidth=2))

legend = g._legend

legend.set_title('Diagnosis')
legend.texts[0].set_text('Benign')
legend.texts[1].set_text('Malignant')


# In[9]:


sns.histplot(x="radius_worst", data=df)


# In[ ]:


#4. Build Model
#Dataset Splitting


# In[10]:


X = df_encoded.drop("diagnosis", axis=1)
y = df_encoded["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


#Scaling


# In[11]:


scale = StandardScaler(copy=True, with_mean=True, with_std=True)
scale.fit(X_train)

X_train_scaled = scale.transform(X_train)
X_test_scaled = scale.transform(X_test)


# In[12]:


print(X_train_scaled.var())


# In[ ]:


#PCA


# In[13]:


print(X_train.var())


# In[14]:


pca1 = PCA(n_components=6, svd_solver='randomized')
pca1.fit(X_train_scaled)

X_train_scaled_pca = pca1.transform(X_train_scaled)
X_test_scaled_pca = pca1.transform(X_test_scaled)


# In[15]:


print(X_train_scaled_pca.var())


# In[ ]:


#SVC Object


# In[16]:


svc = SVC(C=0.9, gamma=0.063, kernel='rbf')
svc.fit(X_train_scaled_pca, y_train)
svc.score(X_train_scaled_pca, y_train), svc.score(X_test_scaled_pca, y_test)


# In[ ]:


#5. Model Evaluation


# In[17]:


preds_svc = svc.predict(X_test_scaled_pca)
report = classification_report(y_test, preds_svc)

print(report)


# In[18]:


cm = confusion_matrix(y_test, preds_svc)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='jet', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[19]:


def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X_train_scaled_pca, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation Score")
    
    plt.legend(loc="best")
    plt.show()

plot_learning_curve(svc, X_train_scaled_pca, y_train)


# In[ ]:




