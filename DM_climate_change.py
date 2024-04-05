#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl 
import numpy as np
import re
import string


# In[2]:


import nltk

nltk.download(["names", "stopwords","state_union","twitter_samples",
               "movie_reviews", "averaged_perceptron_tagger",
               "vader_lexicon","punkt","wordnet"])
nltk.download('omw-1.4')


# ### Read the Climate change data

# In[3]:


df = pd.read_csv('/Users/jyotismac/Desktop/Climate change.csv')


# # I. Data Exploration

# In[4]:


# check first few observations
df.head()


# In[5]:


# check last few observations
df.tail()


# In[6]:


# check shape of data i.e number of records and columns
df.shape


# In[7]:


#check if any missing value in any variable
df.isnull().any()


# In[8]:


# check the data type of each column, number of columns, memory usage, 
# and the number of records in the dataset

df.info()


# In[9]:


#identifying missing value in specific variables as observed in previous step

likes_nan_count=df['Likes'].isna().sum()
retweets_nan_count=df['Retweets'].isna().sum()

print("The number of missing values in Likes column is: " + str(likes_nan_count))
print("The number of missing values in Retweets column is: " + str(retweets_nan_count))


# In[10]:


# Check summary of the dataset
df.describe()


# In[11]:


# check summary of the dataset- categorical variables
# https://stackoverflow.com/questions/64223060/summary-of-categorical-variables-pandas

df.describe(include= 'object')


# # II. DATA PREPROCESSING

# In[12]:


# create a copy of data to extract number of comments from Embedded_text column
df_1 = df.copy()


# In[13]:


# removing all the commas from the embedded text column
df_1['Embedded_text'] = df_1['Embedded_text'].str.replace(',', '')


# In[14]:


# Extract the comment numbers and create new columns
# code source and modified - https://pandas.pydata.org/docs/reference/api/pandas.Series.str.extract.html
df_1[['comments', 'likes', 'retweets']] = df_1['Embedded_text'].str.extract(r'(\d{1,5}(?:[,\.]\d{3})*)(?:\s+)(\d{1,5}(?:[,\.]\d{5})*)(?:\s+)([\d\.]+[k]?)', expand=True)

df_1.head()


# In[15]:


# attaching comments column to original dataset

df['comments']=df_1['comments']
df.head()


# In[16]:


# replacing null values in comments column with 0

df['comments'] = df['comments'].fillna(0).astype(int)


# In[17]:


# creating duplicate data set to clean it

df_clean = df.copy()


# In[18]:


df_clean.isnull().any()


# In[19]:


import contractions

# applying contractions to expand the contracted words
# Code source & modified- https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/

df_clean['Embedded_text'] = df['Embedded_text'].apply(lambda x: contractions.fix(x))
df_clean['Embedded_text']


# In[20]:


# Dropping null values in Likes column 

df_clean = df_clean.dropna(axis = 0,subset = ['Likes']) 


# In[21]:


# Dropping null values in Retweets column 

df_clean = df_clean.dropna(axis = 0,subset = ['Retweets']) 


# In[22]:


# creating instance for storing stopwords of english language

stopwords = nltk.corpus.stopwords.words("english")


# In[23]:


# removing commas from Likes column

df_clean['Likes'] = df_clean['Likes'].str.replace(',', '')


# In[24]:


# removing commas from Retweets column

df_clean['Retweets'] = df_clean['Retweets'].str.replace(',', '')


# In[25]:


# creating a function to change K into 1000 for likes and retweets columns
# code source & modify- https://gist.github.com/gajeshbhat/67a3db79a6aecd1db42343190f9a2f17
def convert_k(val):
    if val[-1] == 'K':
        return int(float(val[:-1])) * 1000
    else:
        return int(float(val))


# In[26]:


# applying above function to Likes and Retweets column and creating two new columns

df_clean['Likes_new'] = df_clean['Likes'].apply(convert_k)
df_clean['Retweets_new'] = df_clean['Retweets'].apply(convert_k)


# In[27]:


#check applicability of convert_k function

df_clean['Retweets_new']


# In[28]:


#check applicability of convert_k function
df_clean['Likes_new']


# In[29]:


# dropping original likes and retweets columns

df_clean = df_clean.drop(['Likes', 'Retweets'], axis=1)


# In[30]:


# removing outliers in comments column

df_clean = df_clean[df_clean['comments'] <= 6500]


# In[31]:


# Define a function to clean and preprocess a tweet

def clean_tweet(tweet):
   
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Remove @mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove #hashtags
    tweet = re.sub(r'#\w+', '', tweet)
    
    # Remove punctuation !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~, string.punctuation contains all these special characters
    ## str.maketrans() creates a table that maps each punctuation character to None
    ## translate() then removes the corresponding characters
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers from the end of each tweet. $ represents end of the text
    tweet = re.sub(r'\s*\d+[k]?\s*\d+[k]?\s*\d+[k]?\s*$','', tweet)
    
    # Tokenize words
    words = tweet.split()
    
    # Remove stop words
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stopwords]
    
    # Lemmatize words- change the words to their stem/root form for better analysis
    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a string
    tweet = ' '.join(words)
    
    return tweet


# In[32]:


# Apply the clean_tweet() function to the embedded_text column

df_clean['clean_text'] = df_clean['Embedded_text'].apply(clean_tweet)


# In[33]:


df_clean['clean_text']


# In[34]:


##length of messages- character length

df_clean['length'] = df_clean['clean_text'].apply(len)
df_clean['length']
df_clean.head()


# In[35]:


# length of longest tweet

max_length = max(df_clean['length'])
print("Length of longest text in 'clean_text' column:", max_length)


# In[36]:


# length of shortest tweet

min_length = min(df_clean['length'])
print("Length of shortest text in 'clean_text' column:", min_length)


# In[37]:


# create a list of all tweets 

words = df_clean['clean_text'].tolist()


# In[38]:


# Remove apostrophes from the words

words = [re.sub(r"['‘’`]", "", word) for word in words]


# In[39]:


# joining all the tweets

words = ' '.join(words)


# In[40]:


# Tokenize the words

words_token = nltk.word_tokenize(words)


# In[41]:


# Extract the text of the tweet from the DataFrame

text = df_clean['clean_text']

# Split the text into a list of words
words_1 = text.str.split()

# Count the number of words in each tweet
word_counts = words_1.apply(len)

# Add the word count as a new column in the DataFrame
df_clean['word_count'] = word_counts

# find the highest number of words in a tweet
highest_word_count = df_clean['word_count'].max()

# print the highest word count
print("Highest word count:", highest_word_count)

# find the lowest number of words in a tweet
lowest_word_count = df_clean['word_count'].min()

# print the highest word count
print("Lowest word count:", lowest_word_count)


# In[42]:


# find all tweets with the less than 5 words
low_word_count = len(df_clean[df_clean['word_count'] < 5])

print("Number of tweets with less than 5 words", low_word_count)


# In[43]:


# creating frequency distribution

fd = nltk.FreqDist(words_token)

# display top 3 most common words used
fd.most_common(3)


# In[44]:


# display top three most common words used in tabular form
fd.tabulate(3)


# ### Two word combinations

# In[45]:


bi_finder = nltk.collocations.BigramCollocationFinder.from_words(words_token)


# In[46]:


bi_finder.ngram_fd.most_common(3)


# ### Three word combinations

# In[47]:


tri_finder = nltk.collocations.TrigramCollocationFinder.from_words(words_token)


# In[48]:


tri_finder.ngram_fd.most_common(3)


# ### Four word combinations

# In[49]:


quad_finder = nltk.collocations.QuadgramCollocationFinder.from_words(words_token)


# In[50]:


quad_finder.ngram_fd.most_common(3)


# # III. STATISTICAL OBSERVATION

# ### Average Counts

# In[51]:


# average number of comments

avg_comments = np.mean(pd.to_numeric(df_clean['comments']))
print("The average number of comments is: " + str(avg_comments))


# In[52]:


# average likes

avg_likes = np.mean(df_clean['Likes_new'])
print("The average number of likes is: " + str(avg_likes))


# In[53]:


# average retweets

avg_retweets = np.mean(df_clean['Retweets_new'])
print("The average number of retweets is: " + str(avg_retweets))


# In[54]:


# Create a bar chart

plt.bar(['Average Likes', 'Average Retweets', 'Average Comments'], [avg_likes, avg_retweets, avg_comments])
plt.title('Average Likes, Retweets and Comments')
plt.xlabel('Metrics')
plt.ylabel('Count')
plt.show()


# ### Frequency Count over period of time

# In[55]:


# monthly frequency of tweets

freq_by_month = df_clean.groupby(pd.to_datetime(df_clean['Timestamp']).dt.month).count()

# change month numbers to three letter month names
freq_by_month.index = pd.to_datetime(freq_by_month.index, format='%m').strftime('%b')
freq_by_month


# In[56]:


# bar chart of monthly frequency of tweets

freq_by_month.plot(kind='bar', y=['clean_text'], rot=0, color= 'pink')

plt.title('Count by month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(['Tweets'])
plt.show()


# In[57]:


# weekly frequency of tweets

freq_by_weekday = df_clean.groupby(pd.to_datetime(df_clean['Timestamp']).dt.weekday).count()

# change weekday numbers to three letter weekday names
freq_by_weekday.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


freq_by_weekday


# In[58]:


# daily frequency of tweets

freq_by_day = df_clean.groupby(pd.to_datetime(df_clean['Timestamp']).dt.day).count()

freq_by_day


# In[59]:


# hourly frequency of tweets

freq_by_hour = df_clean.groupby(pd.to_datetime(df_clean['Timestamp']).dt.hour).count()

freq_by_hour


# In[60]:


# Create line plots of frequency of tweets on monthly, weekday, daily and hourly basis

# Create subplots with 2 rows and 2 column
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot monthly frequency in first subplot
axs[0,0].plot(freq_by_month.index, freq_by_month)
axs[0,0].set_ylabel('Count of tweets')
axs[0,0].set_xlabel('Months')
axs[0,0].set_title('Monthly frequency of tweets')

# Plot weekday frequency in second subplot
axs[0,1].plot(freq_by_weekday.index, freq_by_weekday)
axs[0,1].set_ylabel('Count of tweets')
axs[0,1].set_xlabel('Days of the week')
axs[0,1].set_title('Frequency of tweets on Weekday Basis')

# Plot daily frequency in third subplot
axs[1,0].plot(freq_by_day.index, freq_by_day)
axs[1,0].set_ylabel('Count of tweets')
axs[1,0].set_xlabel('Days of a month')
axs[1,0].set_title('Daily frequency of tweets')

# Plot hourly frequency in third subplot
axs[1,1].plot(freq_by_hour.index, freq_by_hour)
axs[1,1].set_ylabel('Count of tweets')
axs[1,1].set_xlabel('Hours of a day')
axs[1,1].set_title('Hourly frequency of tweets')

# Add common x-axis label
fig.text(0.5, 0.01, 'Time Period', ha='center')

# Add common title
fig.suptitle('Tweet frequency Monthly, Weekday, and Daily Basis')

fig.tight_layout()

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.3)

# Show the plot
plt.show()


# ### Average counts over period of time

# In[61]:


# Average number of likes and retweets on monthly basis

monthly_avg = df_clean.assign(timestamp=pd.to_datetime(df_clean['Timestamp'])).groupby(pd.Grouper(key='timestamp', freq='M'))[['Likes_new', 'Retweets_new']].mean()

# Format the timestamp column to show only the month in three letter format
monthly_avg.index = monthly_avg.index.strftime('%b')

# Rename the index column to 'Month'
monthly_avg.index.name = 'Month'
monthly_avg


# In[62]:


# Average number of likes and retweets on days of the week basis

# Group by day of the week and calculate mean
weekday_avg = df_clean.groupby(pd.to_datetime(df_clean['Timestamp']).dt.dayofweek).mean()[['Likes_new', 'Retweets_new']]

# change weekday numbers to three letter weekday names
weekday_avg.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Rename the index column to 'Weekday'
weekday_avg.index.name = 'Weekday'

weekday_avg


# In[63]:


# Average number of likes and retweets on days of month basis

# Group by days and calculate mean
daily_avg = df_clean.groupby(pd.to_datetime(df_clean['Timestamp']).dt.day).mean()[['Likes_new', 'Retweets_new']]

# change weekday numbers to three letter weekday names
#weekday_avg.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Rename the index column to 'Weekday'
daily_avg.index.name = 'Days'

daily_avg


# In[64]:


# Average number of likes and retweets on hourly basis

# Group by days and calculate mean
hourly_avg = df_clean.groupby(pd.to_datetime(df_clean['Timestamp']).dt.hour).mean()[['Likes_new', 'Retweets_new']]

# change weekday numbers to three letter weekday names
#weekday_avg.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Rename the index column to 'Weekday'
hourly_avg.index.name = 'Days'

hourly_avg


# In[65]:


# Code source for plotting multiple plots in one outcome- https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# Create subplots with three rows and one column
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot monthly average in first subplot
axs[0,0].plot(monthly_avg.index, monthly_avg)
axs[0,0].set_ylabel('Avg Likes & Retweets')
axs[0,0].set_xlabel('Months')
axs[0,0].set_title('Average Likes & Retweets on Monthly Basis')

# Plot weekday average in second subplot
axs[0,1].plot(weekday_avg.index, weekday_avg)
axs[0,1].set_ylabel('Avg Likes & Retweets')
axs[0,1].set_xlabel('Days of the week')
axs[0,1].set_title('Average Likes & Retweets on Weekday Basis')

# Plot daily average in third subplot
axs[1,0].plot(daily_avg.index, daily_avg)
axs[1,0].set_ylabel('Avg Likes & Retweets')
axs[1,0].set_xlabel('Days of a month')
axs[1,0].set_title('Average Likes & Retweets on Daily Basis')

# Plot hourly average in third subplot
axs[1,1].plot(hourly_avg.index, hourly_avg)
axs[1,1].set_ylabel('Avg Likes & Retweets')
axs[1,1].set_xlabel('Hours of a day')
axs[1,1].set_title('Average Likes & Retweets on Hourly Basis')

# Add common x-axis label
fig.text(0.5, 0.01, 'Time Period', ha='center')

# Add common title
fig.suptitle('Average Likes on Monthly, Weekday, and Daily Basis')

fig.tight_layout()

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.3)

# Show the plot
plt.show()


# # IV. EXPLORATORY TASKS

# ## Sentiment Analysis

# In[66]:


# import library
from nltk.sentiment import SentimentIntensityAnalyzer


# In[67]:


# Create an instance of the SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


# In[68]:


# Define a function to classify the sentiment of a tweet
# Code source- https://www.kaggle.com/code/nicolemeinie/sentiment-analysis-twitter-on-climate-change

def sentiment_class(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] > 0:
        return 'positive'
    elif sentiment['compound'] < 0:
        return 'negative'
    else:
       return 'neutral'
    return sentiment['compound']


# In[69]:


# Apply the function to clean_text column of dataset

df_clean['sentiment']= df_clean['clean_text'].apply(sentiment_class)


# In[70]:


# Define a function to apply the sentiment analyzer to each tweet

def get_sentiment_score(text):
    # Apply the sentiment analyzer to the text
    sentiment_scores = sia.polarity_scores(text)
    # Return the compound score, which ranges from -1 (most negative) to 1 (most positive)
    return sentiment_scores['compound']


# In[71]:


# Apply the get_sentiment_score function to the 'Cleaned_text' column 
# and store the result in a new 'Sentiment_score' column

df_clean['Sentiment_score'] = df_clean['clean_text'].apply(get_sentiment_score)


# In[72]:


# Create histogram

plt.hist(df_clean['sentiment'], color='pink', align='mid')

# Get bin counts and centers
counts, bins, _ = plt.hist(df_clean['sentiment'], color= 'green', align='mid')
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Add counts to the center of each bin
for count, x in zip(counts, bin_centers):
    plt.text(x, count, str(int(count)), ha='center', va='bottom')

# Add titles and labels
plt.title('Histogram showing sentiment class counts')
plt.xlabel('Sentiment Class')
plt.ylabel('Count of tweets')

# Show the plot
plt.show()


# In[73]:


# sentiment count

sentiment_counts = df_clean.groupby('sentiment')['sentiment'].count()
print(sentiment_counts)


# In[74]:


# create pie chart with percentages

plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title('Sentiment Counts')
plt.show()


# In[75]:


# Define a function to apply the sentiment analyzer to each tweet to get sentiment score

def get_sentiment_score(text):
    # Apply the sentiment analyzer to the text
    sentiment_scores = sia.polarity_scores(text)
    # Return the compound score, which ranges from -1 (most negative) to 1 (most positive)
    return sentiment_scores['compound']


# In[76]:


# Apply the get_sentiment_score function to the 'clean_text' column and store the result in a new 'Sentiment_score' column

df_clean['Sentiment_score'] = df_clean['clean_text'].apply(get_sentiment_score)


# In[77]:


#  visualize the sentiment scores

plt.hist(df_clean['Sentiment_score'])
plt.show()


# ## Wordcloud

# In[78]:


from wordcloud import WordCloud
# Code source-  https://www.datacamp.com/tutorial/wordcloud-python

# joining all the words in all tweets
all_words = " ".join([sentence for sentence in df_clean['clean_text']])

# create a wordcloud for all words
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100, background_color='white').generate(all_words)

# plot the wordcloud
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[79]:


from nltk.probability import FreqDist

# create frequency distribution of all words in clean_text
all_words = " ".join([sentence for sentence in df_clean['clean_text']])
words_list = all_words.split()
freq_dist = FreqDist(words_list)

# select top 200 most frequent words
top_words = pd.DataFrame(freq_dist.most_common(200), columns=['Word', 'Frequency'])
top_words = top_words.set_index('Word')

# generate word cloud for top 200 words
wordcloud_1 = WordCloud(width = 800, height = 800, background_color ='white', 
                min_font_size = 10).generate_from_frequencies(top_words.to_dict()['Frequency'])

# plot the wordcloud
plt.figure(figsize=(15,8))
plt.imshow(wordcloud_1, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[80]:


# hashtag word cloud
# code source- https://www.kaggle.com/code/nicolemeinie/sentiment-analysis-twitter-on-climate-change

# Extract hashtags words from the 'embedded_text' column
hashtags = []
for text in df_clean['Embedded_text']:
    hashtags += re.findall(r'#(\w+)', text)

# Join all hashtags words into a single string
hashtags_text = ' '.join(hashtags)

# Generate word cloud
word_cloud_2 = WordCloud(background_color='white', max_words=50).generate(hashtags_text)

# Display the word cloud
plt.imshow(word_cloud_2, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[81]:


# sentiment- positive negative neutral word cloud
# code source & modify- https://medium.com/codex/making-wordcloud-of-tweets-using-python-ca114b7a4ef4

# Define a function to generate word cloud for a given sentiment class

def generate_wordcloud(sentiment_class):
    
    # Combine all tweets of the given sentiment class into a single string
    tweets = df_clean[df_clean['sentiment'] == 'positive']['clean_text'].str.cat(sep=' ')

    # Generate the word cloud
    word_cloud_pos = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords).generate(tweets)

    # Plot the word cloud
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(word_cloud_pos)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title(f'Word Cloud for {sentiment_class.capitalize()} Tweets')
    plt.show()

# Generate word cloud for each sentiment class

for sentiment in df_clean['sentiment'].unique():
    generate_wordcloud(sentiment)


# # V. MACHINE LEARNING MODELS

# In[82]:


# create a copy to run machine learning models

df_ml = df_clean.copy()


# ## Feature extraction

# ### Extracting URLs

# In[83]:


# add a new column to the dataframe to store if a tweet contains URL or not
# code source & modify- https://developers.google.com/edu/python/regular-expressions

url_pattern = r'(https?://\S+)'
df_ml['has_urls'] = df_ml['Embedded_text'].apply(lambda x: 1 if re.findall(url_pattern, x) else 0)


# In[84]:


# extract counts of URLs in each row of the 'Embedded_text' column

df_ml['url_count'] = df_ml['Embedded_text'].apply(lambda x: len(re.findall(url_pattern, x)))


# ### Extracting Images & Videos

# In[85]:


# Define function to check if each tweet contains an image

def contains_image(row):
    text = row['Embedded_text']
    if 'pic.twitter.com' in text:
        # the tweet contains an image
        return True
    else:
        # The tweet does not contain an image
        return False


# In[86]:


# Add a new column to the dataframe to indicate if each tweet contains an image

df_ml['contains_image'] = df_ml.apply(contains_image, axis = 1)


# In[87]:


# define function to check if tweet contains images and return count

def count_images(tweet):
    images = tweet.split()
    count = sum(image.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) for image in images)
    return count

# extract count of images in each tweet
df_ml['image_count'] = df_ml['Embedded_text'].apply(count_images)


# In[88]:


# define function to check if tweet contains images and return count

def count_videos(tweet):
    videos = tweet.split()
    count = sum(video.lower().endswith('.mp4') for video in videos)
    return count

# extract count of videos in each tweet
df_ml['video_count'] = df_ml['Embedded_text'].apply(count_videos)


# ### Extracting Hashtags

# In[89]:


# Define function to check if each tweet contains a hashtag
# code source & modify - https://www.kaggle.com/code/nicolemeinie/sentiment-analysis-twitter-on-climate-change

def contains_hashtag(row):
    text = row['Embedded_text']
    if re.search(r'\#\w+', text):
        # the tweet contains a hashtag
        return 1
    else:
        # The tweet does not contain a hashtag
        return 0


# In[90]:


# Apply the function to the dataframe and create a new column

df_ml['contains_hashtag'] = df_ml.apply(contains_hashtag, axis=1)


# In[91]:


# Define function to count number of hashtags in each tweet

def count_hashtags(row):
    text = row['Embedded_text']
    hashtags = re.findall(r'#\w+', text)
    return len(hashtags)


# In[92]:


# Apply the function to the dataframe and create a new column for count of hashtags

df_ml['hashtag_count'] = df_ml.apply(count_hashtags, axis=1)


# ### Extracting words count 

# In[93]:


# create new column with word count

df_ml['word_count'] = df_ml['clean_text'].apply(lambda x: len(x.split()))


# ### Extracting time (hour, day of week) of tweets

# In[94]:


# code source & Modify- https://pynative.com/python-get-the-day-of-week/

# Convert the timestamp column to datetime format
df_ml['timestamp'] = pd.to_datetime(df_ml['Timestamp'])

# Extract the hour from the datetime column
df_ml['hour'] = df_ml['timestamp'].dt.hour

# Extract the weekday from the datetime column
df_ml['weekday'] = df_ml['timestamp'].dt.weekday

# Define the time slots for morning, afternoon, evening, and night
time_slots = [0, 6, 12, 18, 24]
time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']

# Categorize the hours into the defined time slots
df_ml['time_slot'] = pd.cut(df_ml['hour'], bins=time_slots, labels=time_labels, include_lowest=True)

# drop timestamp column
df_ml = df_ml.drop(['timestamp'], axis=1)


# In[95]:


#df_ml['comments'] = df_ml['comments'].fillna(0)
#df_ml['comments'] = pd.to_numeric(df_ml['comments'], errors='coerce')
#df_ml['time_slot'] = pd.to_numeric(df_ml['time_slot'], errors='coerce')
# df_ml['sentiment_score'] = pd.to_numeric(df_ml['sentiment_score'], errors='coerce')
#df_ml['length'] = pd.to_numeric(df_ml['length'], errors='coerce')


# ## MODEL 1: LINEAR REGRESSION

# In[127]:


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Create TF-IDF matrix of tweet text
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_ml['Embedded_text'])

# Create feature matrix for sentiment scores and hashtags
# Define the feature columns
features = ['weekday', 'hour','Retweets_new','Sentiment_score', 'hashtag_count','url_count','image_count','video_count','comments', 'word_count']
X_features = pd.DataFrame(df_ml[features])

# Concatenate feature matrix with TF-IDF matrix
X = hstack([tfidf_matrix, X_features.values])

# Define target variable
y = df_ml['Likes_new']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Train Linear Regression model
lr = LinearRegression()
lr1 = lr.fit(X_train, y_train)

# Predict likes on test set
y_pred = lr.predict(X_test)

# Calculate R-squared
r2_lr = r2_score(y_test, y_pred)

# Calculate the mean squared error
mse_lr = mean_squared_error(y_test, y_pred)

# Calculate the root mean squared error
rmse_lr = mean_squared_error(y_test, y_pred, squared=False)

# Calculate the mean absolute error
mae_lr = mean_absolute_error(y_test, y_pred)

# Get the coefficients of the model
coef_lr_linear = lr.coef_

# Calculate model accuracy
accuracy_lr = lr.score(X_test, y_test)

print('Model 1: LINEAR REGRESSION', '\n')
print("R-squared score:", r2_lr)
print("Mean squared error:", mse_lr)
print("Root mean squared error:", rmse_lr)
print("Mean absolute error:", mae_lr)
print("Accuracy of model: ", accuracy_lr, '\n')

print('coefficients are : ')

# Get the coefficients of the model
intercept_lr1 = lr1.intercept_
coefs_lr1 = list(lr1.coef_)

# Print the intercept and coefficients with their names- not including in printed report 
# as coefficients of each word made the report 700 pages long.
#print('Intercept:', intercept_lr1)
#for feature, coefficient in zip(features + tfidf.get_feature_names(), coefs_lr1):
    #print(f"{feature}: {coefficient}")


# ### Cross Validation of Linear Regreesion Model

# In[108]:


# Cross validation using K-fold cross validation method

from sklearn.model_selection import cross_val_score, KFold

# Set the number of folds
num_folds = 10

X = X #independent variables
Y = y #response variable

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation on the lr model

cv_lr = cross_val_score(lr, X, y, cv=kfold)


# Calculate the mean and standard deviation of the scores
print('Average validation score of Linear Regression model is:', cv_lr.mean())
print('The Standard deviation of Linear Regression is:', cv_lr.std())


# ## MODEL 2: XG BOOST

# In[123]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Convert the data to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the hyperparameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror'
}

# Train the model
num_rounds = 100
model_xgb = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
y_pred_xgb = model_xgb.predict(dtest)

# Calculate R-squared
r2_xgb = r2_score(y_test, y_pred_xgb)

# Calculate the mean squared error
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# Calculate the root mean squared error
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared = False)

# Calculate the mean absolute error
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print('Model 2: XG BOOST', '\n')
print("R-squared score:", r2_xgb)
print("Mean squared error:", mse_xgb)
print("Root mean squared error:", rmse_xgb)
print("Mean absolute error:", mae_xgb)
print("Accuracy of model: ", r2_xgb, '\n')

#print('coefficients are : ')

# Get the feature importance scores of the model
#importance_scores = model_xgb.get_score(importance_type='weight')

# Print the feature importance scores with their names
#for feature, score in importance_scores.items():
 #   print(f"{feature}: {score}")


# ### Cross Validation of XG Boost Model

# In[124]:


# Cross validation using K-fold cross validation method

# Set the number of folds
num_folds = 10

X = X #independent variables
Y = y #response variable

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# creating xgb classifier
xgbm = xgb.XGBRegressor()

# Perform K-fold cross-validation on the lr model

cv_xgb = cross_val_score(xgbm, X, y, cv=kfold)

# Calculate the mean and standard deviation of the scores
print('Average validation score of XGBoost model is:', cv_xgb.mean())
print('The Standard deviation of XGBoost model is:', cv_xgb.std())


# ## MODEL 3: RANDOM FOREST REGRESSOR

# In[122]:


from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=123)
rf.fit(X_train, y_train)

# Evaluate the model
train_score = rf.score(X_train, y_train)
test_score = rf.score(X_test, y_test)
print('Training score:', train_score)
print('Testing score:', test_score)

# Make predictions on the test set
y_pred_rf = rf.predict(X_test)

# Calculate R-squared
r2_rf = r2_score(y_test, y_pred_rf)

print('R-squared:', r2_rf)

# Calculate the mean squared error
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Calculate the root mean squared error
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

# Calculate the mean absolute error
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Get the coefficients of the model
coef_rf_linear = lr.coef_

print("Coefficients:", coef_rf_linear)

# Calculate model accuracy
accuracy_rf = rf.score(X_test, y_test)

print("R-squared score:", r2_rf)
print("Mean squared error:", mse_rf)
print("Root mean squared error:", rmse_rf)
print("Mean absolute error:", mae_rf)
print("Accuracy of model: ", accuracy_rf)


# ### Cross Validation of Random Forest Regressor Model

# In[ ]:


# Cross validation using K-fold cross validation method

# Set the number of folds
num_folds = 10

X = X #independent variables
Y = y #response variable

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation on the lr model

cv_rf = cross_val_score(rf, X, y, cv=kfold)

# Calculate the mean and standard deviation of the scores
print('Average validation score of XGBoost model is:', cv_rf.mean())
print('The Standard deviation of XGBoost model is:', cv_rf.std())


# ### Comparison of three models

# In[126]:


# COde source & modify- https://realpython.com/how-to-use-numpy-arange/
# storing performance metrices of three models in objects
models = ['lr', 'model_xgb', 'rf']
rmse = [rmse_lr, rmse_xgb, rmse_rf]
mae = [mae_lr, mae_xgb, mae_rf]
r2 = [r2_lr, r2_xgb, r2_rf]

# Define the bar width
barWidth = 0.25

# Set the position of the bars on the x-axis
r1 = np.arange(len(r2))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create the bar graph
plt.bar(r1, r2, color='red', width=barWidth, edgecolor='white', label='R-squared')
plt.bar(r2, rmse, color='green', width=barWidth, edgecolor='white', label='RMSE')
plt.bar(r3, mae, color='blue', width=barWidth, edgecolor='white', label='MAE')

# Add xticks on the middle of the group bars
plt.xlabel('Models')
plt.xticks([r + barWidth for r in range(len(r2))], ['Linear Reg', 'XG Boost', 'SVR'])


# Set the y-axis label
plt.ylabel('Scores')

# Add a legend
plt.legend()

# Show the plot
plt.show()

