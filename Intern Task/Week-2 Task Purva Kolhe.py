#!/usr/bin/env python
# coding: utf-8

# ## Name:- Purva Deepak Kolhe
# ## Qualification: pursuing M.Sc(Data Science)
# ## Contact No.: 8329201092
# ## Email:- kolhepurva01@gmail.com
# ## Candidate No.: AB-Tech 0052

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("D:\\FC practical work\\Machine Learning\\books.csv",error_bad_lines = False)


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.corr()


# In[14]:





# In[ ]:





# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated()


# ### 1. What is the most popular book?

# In[8]:


most_popular_book = df[df['average_rating'] == df['average_rating'].max()]['title'].values[0]
print("The most popular book is:", most_popular_book)


# ### 2. Are books with fewer pages rated higher than those with large page counts

# In[9]:


# Books with fewer pages (less than 500 pages)
fewer_pages_ratings = df[df['num_pages'] < 500]['average_rating']

# Books with larger page counts (500 pages or more)
larger_pages_ratings = df[df['num_pages'] >= 500]['average_rating']

if fewer_pages_ratings.mean() > larger_pages_ratings.mean():
    print("Books with fewer pages are rated higher.")
else:
    print("Books with larger page counts are rated higher.")


# ### 3. What is the most popular book of the 60s?

# In[10]:


books_60s = df[(df['publication_date'] > '1960') & (df['publication_date'] < '1969')]
most_popular_60s = books_60s[books_60s['average_rating'] == books_60s['average_rating'].max()]['title'].values[0]
print("The most popular book of the 60s is:", most_popular_60s)


# ### 4. Who wrote the most pages?

# In[11]:


author_most_pages = df.groupby('authors')['num_pages'].sum().idxmax()
print("The author who wrote the most pages is:", author_most_pages)


# ### 5. What's an author's average page count?

# In[ ]:


authors_avg_page_count = df.groupby('authors')['num_pages'].mean()
print(authors_avg_page_count)


# ### 6. How many books have been written with less than 200 pages?

# In[ ]:


num_books_less_than_200_pages = df[df['num_pages'] < 200].shape[0]
print("Number of books with less than 200 pages:", num_books_less_than_200_pages)


# ### 7. What is Houghton Mifflin Harcourt's most popular book?

# In[ ]:


houghton_mifflin_books = df[df['authors'] == 'Houghton Mifflin Harcourt']
houghton_mifflin_most_popular = houghton_mifflin_books[houghton_mifflin_books['average_rating'] == houghton_mifflin_books['average_rating'].max()]['title'].values[0]
print("Houghton Mifflin Harcourt's most popular book is:", houghton_mifflin_most_popular)


# ### 8. Display the most popular book written by each author.

# In[ ]:


# most_popular_books_by_author = df.groupby('authors')['average_rating'].idxmax()
popular_books_df = df.loc[most_popular_books_by_author, ['authors', 'title', 'average_rating']]
print("Most popular book written by each author:")
print(popular_books_df)


# ### 9. What is the least popular book of the 90s?

# In[ ]:


books_90s = df[(df['publication_year'] >= 1990) & (df['publication_year'] <= 1999)]
least_popular_90s = books_90s[books_90s['average_rating'] == books_90s['average_rating'].min()]['title'].values[0]
print("The least popular book of the 90s is:", least_popular_90s)


# ### 10. What is the highest-rated book with over 500 pages?

# In[ ]:


books_over_500_pages = df[df['num_pages'] > 500]
highest_rated_over_500_pages = books_over_500_pages[books_over_500_pages['average_rating'] == books_over_500_pages['average_rating'].max()]['title'].values[0]
print("The highest-rated book with over 500 pages is:", highest_rated_over_500_pages)


# In[ ]:





# # Descriptive Statistics

# ### 1. What are typical values in this dataset?

# In[15]:


typical_values = df[['average_rating', 'num_pages', 'ratings_count']].describe().loc[['mean', '50%']]
print("Typical values in the dataset:")
print(typical_values)


# ### 2. How widely do values in the dataset vary?(st.dv.)

# In[ ]:


variability = df[['average_rating', 'num_pages', 'ratings_count']].std()
print("Variability (standard deviation) in the dataset:")
print(variability)


# ### 3. Are there any unusually high or low values in this dataset? (outliers)

# In[ ]:


import matplotlib.pyplot as plt

# Box plots of numeric columns
df[['average_rating', 'num_pages', 'ratings_count']].boxplot()
plt.title("Box plots of numeric columns")
plt.show()


# ### 4. What is the size of the dataset?

# In[ ]:


dataset_size = df.shape
print("Size of the dataset:", dataset_size)


# ### 5. Is any observation repeating more frequently than others?

# In[ ]:


frequent_observations = df.mode().loc[0]
print("Most frequent observations:")
print(frequent_observations)


# ### 6. What is the central value of different columns?

# In[ ]:


central_values = df[['average_rating', 'num_pages', 'ratings_count']].mean()
print("Central values of different columns:")
print(central_values)


# ### 7. What is the most popular book?
# 

# In[ ]:


most_popular_book = df[df['average_rating'] == df['average_rating'].max()]['title'].values[0]
print("The most popular book is:", most_popular_book)


# ### 8. Create some storyline and relevant graphs to present your findings from the analysis
# that you have performed.

# In[ ]:


# Histogram of average ratings
plt.hist(df['average_rating'], bins=20)
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings')
plt.show()


# In[ ]:




