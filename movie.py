#!/usr/bin/env python
# coding: utf-8

# In[196]:


from sklearn.feature_extraction.text  import CountVectorizer


# In[197]:


text=['London Paris London',"Paris Paris London"]
cv=CountVectorizer()
count_mat=cv.fit_transform(text)
print(count_mat.toarray())


# In[198]:


from sklearn.metrics.pairwise import cosine_similarity


# In[199]:


similarity_score=cosine_similarity(count_mat)


# In[200]:


print(similarity_score)


# In[218]:


def get_language_from_director_name(director_name):
    return df[df.director_name == director_name]["language"]

def get_director_name_from_language(language):
    return df[df.language == language]["director_name"]


# In[202]:


import pandas as pd


# In[203]:


df=pd.read_csv("movie_metadata.csv")


# In[204]:


df


# In[205]:


features = ['director_name','genres','language','actor_2_name']


# In[206]:


for feature in features:
     df[feature] = df[feature].fillna(' ')


# In[ ]:





# In[207]:


features


# In[ ]:





# In[208]:


def combine_features(row):
    try:
        return row['director_name']+" "+row["genres"]+" "+row["language"]+" "+row["actor_2_name"]
    except:
        print("Error:", row)

df["combined_features"] = df.apply(combine_features,axis=1)

print("Combined Features:", df["combined_features"].head())


# In[209]:


cv = CountVectorizer()


# In[210]:


count_matrix = cv.fit_transform(df["combined_features"])


# In[211]:


count_matrix


# In[212]:


cosine_sim = cosine_similarity(count_matrix) 


# In[219]:


movie_user_likes ="James Cameron"


# In[220]:


movie_director_name= get_director_name_from_language(movie_user_likes)


# In[222]:


similar_movies =  list(enumerate(cosine_sim[movie_director_name]))


# In[223]:


sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)


# In[228]:


i=0
for element in sorted_similar_movies:
    print(get_director_name_from_language(element[0]))
    i=i+1
    if i>50:
        break


# In[ ]:





# In[ ]:




