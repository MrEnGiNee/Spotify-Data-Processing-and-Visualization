#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df_tracks = pd.read_csv("C:/Users/dell/Downloads/Spotify Datasets/tracks.csv")
df_tracks.head()


# In[7]:


# null values

pd.isnull(df_tracks).sum()


# In[8]:


df_tracks.info()


# In[9]:


sorted_df = df_tracks.sort_values('popularity', ascending = True).head(10)
sorted_df


# In[10]:


df_tracks.describe().transpose()


# In[11]:


most_popular = df_tracks.query('popularity>90', inplace = False).sort_values('popularity', ascending = False)
most_popular[:10]


# In[12]:


df_tracks.set_index("release_date", inplace=True)
df_tracks.index=pd.to_datetime(df_tracks.index)
df_tracks.head()


# In[14]:


df_tracks[["artists"]].iloc[29]


# In[ ]:


df_tracks["duration"]= df_tracks["duration_ms"].apply(lambda x: round(x/1000))
df_tracks.drop("duration_ms", inplace=True, axis=1)


# In[17]:


df_tracks.duration.head()


# In[19]:


corr_df=df_tracks.drop(["key", "mode", "explicit"],axis=1).corr(method="pearson")
plt.figure(figsize=(14,6))
heatmap=sns.heatmap(corr_df,annot=True,fmt=".1g", vmin=-1, vmax=1, center=0, cmap="inferno", linewidth=1, linecolor="Black")
heatmap.set_title("Correlation Heatmap Between Variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)


# In[20]:


sample_df = df_tracks.sample(int(0.004*len(df_tracks)))


# In[21]:


print(len(sample_df))


# In[22]:


plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y = "loudness", x = "energy", color = "c").set(title = "Loudness vs Energy")


# In[23]:


plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y = "popularity", x = "acousticness", color = "b").set(title = "Popularity vs Acousticness")


# In[24]:


df_tracks['dates']=df_tracks.index.get_level_values('release_date')
df_tracks.dates=pd.to_datetime(df_tracks.dates)
years=df_tracks.dates.dt.year


# In[25]:


sns.displot(years,discrete=True,aspect=2,height=5,kind="hist").set(title="Number of songs per year")


# In[27]:


total_dr = df_tracks.duration
fig_dims = (18,7)
fig, ax = plt.subplots(figsize = fig_dims)
fig = sns.barplot(x = years, y = total_dr, ax = ax, errwidth = False).set(title="Year vs Duration")
plt.xticks(rotation=90)


# In[28]:


total_dr=df_tracks.duration
sns.set_style(style="whitegrid")
fig_dims = (10,5)
fig, ax = plt.subplots(figsize=fig_dims)
fig=sns.lineplot(x=years, y=total_dr,ax=ax).set(title="Year vs Duration")
plt.xticks(rotation=75)


# In[29]:


df_genre=pd.read_csv("C:/Users/dell/Downloads/Spotify Datasets/SpotifyFeatures.csv")


# In[31]:


df_genre.head(15)


# In[32]:


plt.title("Duration of the Songs in Different Genres")
sns.color_palette("rocket", as_cmap=True)
sns.barplot(y='genre', x='duration_ms', data=df_genre)
plt.xlabel("Duration in milli seconds")
plt.ylabel("Genres")


# In[33]:


sns.set_style(style = "darkgrid")
plt.figure(figsize=(10,5))
famous = df_genre.sort_values("popularity", ascending = False).head(10)
sns.barplot(y='genre', x='popularity', data = famous).set(title= "Top 5 Genres by Popularity")


# In[ ]:




