# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:25:01 2020

@author: Farhan
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# this dataset was downloaded from https://data.world/data-society/imdb-5000-movie-dataset
movies_dts = pd.read_csv("movie_dataset.csv")

features = ['keywords','cast','genres','director']

def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

for i in features:
    movies_dts[i] = movies_dts[i].fillna('') #filling all NaNs with blank string

movies_dts["combined_features"] = movies_dts.apply(combine_features,axis=1) #applying combined_features() method over each rows of dataframe and storing the combined string in "combined_features" column
movies_dts.iloc[0].combined_features

cv = CountVectorizer() #creating new CountVectorizer() object
count_matrix = cv.fit_transform(movies_dts["combined_features"]) #feeding combined strings(movie contents) to CountVectorizer() object

cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return movies_dts[movies_dts.index == index]["title"].values[0]
def get_index_from_title(title):
    return movies_dts[movies_dts.title == title]["index"].values[0]

movie_user_likes = input("Enter name of the Movie you like most, to see top similar movies like that one | 'Please use a suitable name i.e. [Avatar, Gravity, Wanted, Spider-Man]' : ")

try:
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index])) #accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
    i=0
    print("Top 10 similar movies to "+movie_user_likes+" are:\n")
    for element in sorted_similar_movies:
        print(get_title_from_index(element[0]))
        i+=1
        if i>10:
           break
except:
    print("Please enter a valid movie name.")
