#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None


# In[10]:


df = pd.read_csv('drug_deaths.csv')


# In[11]:


df.head()


# In[12]:


df.info()


# In[18]:





# In[19]:



year = pd.to_datetime(df['Date']).dt.year.value_counts()
plt.figure(figsize = (10, 5))
with plt.style.context('fivethirtyeight'):
    graph1 = sns.barplot(x = year.index.astype('int64'), y = year.values.astype('int64'), 
                         palette=sns.cubehelix_palette(8))
plt.tight_layout()
plt.ylabel('Deaths')
autolabel(graph1)
plt.show()


# In[20]:


age = df['Age']
with plt.style.context('fivethirtyeight'):
    plt.hist(age, bins = range(0, 100, 10), edgecolor = 'black', color = 'tab:purple')
    plt.xticks(range(0, 100, 10))
    plt.xlabel('Age')
    plt.ylabel('Deaths')


# In[21]:



residence_city = df['ResidenceCity'].copy().dropna()
residence_city_cloud = ' '.join(city for city in residence_city)
wc = WordCloud(width=2500, height=1500).generate(residence_city_cloud)
plt.figure(figsize = (10, 8))

plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.margins(x = 0, y = 0)
plt.show()


# In[22]:


drugs = df.loc[::, 'Heroin':'AnyOpioid']
drugs['Fentanyl'] = drugs['Fentanyl'].replace(['1-A', '1 POPS', '1 (PTCH)'], '1')
drugs['AnyOpioid'] = drugs['AnyOpioid'].replace({'N':'0'})
drugs['Morphine_NotHeroin'] = drugs['Morphine_NotHeroin'].replace(['STOLE MEDS', 'PCP NEG', '1ES', 'NO RX BUT STRAWS'], '1')


drugs[['Fentanyl_Analogue', 'Morphine_NotHeroin', 'AnyOpioid', 'Fentanyl']] = drugs[['Fentanyl_Analogue', 'Morphine_NotHeroin', 'AnyOpioid', 'Fentanyl']].astype('int64')
drug = drugs.sum().sort_values(ascending = False).index
frequency = drugs.sum().sort_values(ascending = False).values
plt.figure(figsize = (10, 6))
plt.tight_layout()
with plt.style.context('fivethirtyeight'):
    s = sns.barplot(x = frequency, y = drug, palette=sns.color_palette("Reds_r", 16))
    for patch in s.patches:
        plt.annotate(patch.get_width().astype('int64'), xy = (patch.get_width(), patch.get_y() + patch.get_height()/2),
                    xytext = (1, 0), textcoords = 'offset points', va = 'center', fontsize = 13)
plt.xlabel('Number of times drugs involved in Deaths', fontsize = 13) 


# In[23]:


male = df['Sex'].value_counts().values[0]
female = df['Sex'].value_counts().values[1]
plt.pie([male, female], labels = ['Male', 'Female'], autopct = '%1.1f%%', shadow = True, pctdistance=0.5,
        startangle=90, radius = 1.5, wedgeprops={'edgecolor':'white'}, 
        textprops={'fontweight':'bold', 'fontsize':16})
plt.show()


# In[24]:


df['InjuryPlace'].value_counts().nlargest(5)


# In[25]:


df['MannerofDeath'] = df['MannerofDeath'].replace(['accident', 'ACCIDENT'], 'Accident')
df['MannerofDeath'].value_counts()


# In[ ]:




