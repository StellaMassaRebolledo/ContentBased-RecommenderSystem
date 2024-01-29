# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:32:38 2023

@author: Stella Massa Rebolledo


"""


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

df = pd.read_json("E:\Study_Repository\Centennial College\Fall 2023\COMP 262 - NLP\Assignment #3\meta_Digital_Music.json", lines=True)


'''     ------------   Data Exploration  -----------------   '''

df.shape
df.info()

df.head(5)


# cols = df.columns
cols = df.columns

# # # Replace lists or empty strings by nan
for i in cols:
 df[i] = df[i].apply(lambda x: np.nan if (isinstance(x, (list, str)) and len(x) == 0) else x)

df.info()

'''     ------------   Feature Engineering  -----------------   '''
df2 = df[['description', 'title', 'brand']]


# # Get rid of lists
def clean_description(content):
    if isinstance(content, list):
        return ''.join(content)
    return content

df2['description'] = df2['description'].apply(clean_description)  

# Replacing nan to apply Beatifulsoup parser
df2 = df2.replace(np.nan, "")

# # # BeatifulSoup Parser
cs = ['description', 'title', 'brand']   
for i in cs:
    # df2[i] = df2[i].apply(func_test)
    df2[i] = df2[i].apply(lambda x: BeautifulSoup(x, "lxml").text)

#cleantext = BeautifulSoup(raw_html, "lxml").text
# def func_test(text):
#     print(text)
#     text = BeautifulSoup(text, "lxml").text
#     return text    
    
    
# # # Extract remaining text from Parsing operation
rest_parse = r'\">'   
df2['title'] = df2['title'].str.replace(rest_parse, '', regex=True)
    

# # # Remove special characters for brand and description
remove = r"[^\w\s\d]"
df2['brand'] = df2['brand'].str.replace(remove, "", regex=True)  
df2['description'] = df2['description'].str.replace(remove, "", regex=True)  


# # # Removing leading and trailing spaces
for i in cs:
    df2[i] = df2[i].apply(lambda x: x.strip())


# # # Lowercasing
def lower(text):
    transform = text.lower()
    return transform

df2 = df2.applymap(lower)


# # # Special treatment for title
regex_title=r"[^\w\s\d&]"
df2['title'] = df2['title'].str.replace(regex_title,"", regex=True)

df2['title'] = df2['title'].apply(lambda x: x.strip())

# # # Remove the & at the beginning of the lines
amper = r"^&"
amper2 = r"^&&"
df2['title'] = df2['title'].str.replace(amper, "", regex=True).replace(amper2, "", regex=True)



# # # Dropping duplicates for the three columns
# 74347 --> 65604
df2 = df2.drop_duplicates()


# # # Treatment for multilines in description
# # # Multiline column
df2['description'] = df2['description'].str.replace('\n','')


# # # Trailing and leading spaces
tr_ld =r" ^[ \t]+|[ \t]+$"
df2 = df2.replace(tr_ld, '', regex=True)

def remove_spaces(value):
    if isinstance(value, str) and len(value)<2:
        return str(value).replace(' ', '')
    else:
        return value

df2 = df2.applymap(remove_spaces)



# # #Replacing empty values by nan
df2 = df2.replace('', np.nan)



# # #Dropping rows that have nan values for all the cols
# 65604 --> 65603
df2 = df2.dropna(how='all')

# # # Droping all rows for which title is null
# 65603 --> 60986
df2 = df2.dropna(subset=['title'])

# # # Drop nan values for the columns I will use to find the similarities: description and brand
# 60986 --> 27474
df2 = df2.dropna(subset=['description', 'brand'])

   
df2.info()


# # df2 with new col
df2['description_brand'] = df2['description'] + ' '+ df2['brand']

    

# # #Non english characters
# non_e = r'[^\x00-\x7F]+'
# df2 = df2.replace(non_e, '', regex=True)



# # # Reindexing dataframe to cross-refer the songs' titles
df2 = df2.reset_index()
df2.drop(columns=['index'], inplace=True)


# # # Indices - Title
indices = pd.Series(df2.index, index=df2['title'])

indices['speak to me']


# # # Clean df
df_cleaned = df2[['title', 'description_brand']] 


# # # Instantiating TFIDF object and removing stop words
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df2['description_brand'])

tfidf_matrix.shape


# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim.shape
type(cosine_sim)


l = cosine_sim[1]
print(l)


# # # Saving the cosine similarity matrix
path = "E:\Study_Repository\Centennial College\Fall 2023\COMP 262 - NLP\Assignment #3\cosine_similiarity.npy"

np.save(path, cosine_sim)



# # Loading cosine similarity matrix
cos_sim = np.load(path)


'''     ------------   Recommendation Function  -----------------   '''

# Function that takes in song title as input and gives recommendations 
def content_recommender(title, cosine_sim=cos_sim, df=df_cleaned, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all songs with that title
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first song - it's itself. 
    # cosine similarity for the same song is 1., it will be the highest value.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    song_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return list(df['title'].iloc[song_indices])



'''     ------------   Recommendation Working  -----------------   '''

titles = list(df_cleaned['title'])

while True: 
    user_inpt = input("Please enter a song title: ").lower()
    
    if user_inpt=='exit':
        break

    if user_inpt in titles:
        recommendation = content_recommender(user_inpt)
        print("The top ten recommendations according to your input are: ")
        for i, j in enumerate(recommendation):
            print(i+1,". ", j)
        print('\n')
    else:
        print (f"We don't have any recommendation for {user_inpt}")
        print ('\n')








