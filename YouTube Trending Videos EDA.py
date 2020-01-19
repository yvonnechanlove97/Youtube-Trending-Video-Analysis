# Databricks notebook source
# MAGIC %md
# MAGIC ### Problem statement
# MAGIC YouTube is the most popular video website in the world. They mantain a list of the top trending videos of every day which are selected by their own algorithms based on many factors of a video aside from just the number of views. As a YouTube user, it would be very interesting to find out insights about the trending and hopefully come out patterns that can help YouTubers to be more successful with their channels. Having considered that there is no suitable target variables in the dataset, thus, in this project, I would only focus on data visualization.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data
# MAGIC * The dataset is a csv file conataing a list of the trending YouTube videos from 2017-11 to 2018-06 in the US and Canada including the video title, channel title, category_id, publish time, tags, views, likes and dislikes, description, and comment count.  
# MAGIC * The 'category_id' varies bewteen regions with the specific category in associated JSON files.
# MAGIC 
# MAGIC ### Method
# MAGIC In this project, **Spark** will be used for data preparation and data wrangling, **Python** will be used for data visulization.

# COMMAND ----------

#Import data and drop useless columns
us = spark.read.csv('/FileStore/tables/USvideos.csv', header="true", inferSchema="true").drop('description','comment_disabled','ratings_disabled','video_error_or_removed')
ca = spark.read.csv('/FileStore/tables/CAvideos.csv', header="true", inferSchema="true").drop('description','comment_disabled','ratings_disabled','video_error_or_removed')
us.head()

# COMMAND ----------

# MAGIC %md
# MAGIC In our current dataframe, the category column is in numbers instead of the strings. The according strings are in a nested JSON file. Next, the number category will be converted to the real category name.

# COMMAND ----------

#Insert category column
us_cate=spark.read.json("/FileStore/tables/US_category_id.json",multiLine=True)
ca_cate=spark.read.json("/FileStore/tables/CA_category_id.json",multiLine=True)

from pyspark.sql.functions import explode
from pyspark.sql.functions import monotonically_increasing_id

#US area category
us_cate_df=us_cate.select('items.id',"items.snippet.title")
us_id=us_cate_df.select(explode("id").alias("id"))
us_title=us_cate_df.select(explode("title").alias("category"))
us_id = us_id.withColumn("index", monotonically_increasing_id())
us_title = us_title.withColumn("index", monotonically_increasing_id())
df1 = us_title.join(us_id, "index", "outer").drop("index")
us=us.withColumnRenamed('category_id', 'id')
us=us.join(df1,'id',how='left').drop('id')

#CA area category
ca_cate_df=ca_cate.select('items.id',"items.snippet.title")
ca_id=ca_cate_df.select(explode("id").alias("id"))
ca_title=ca_cate_df.select(explode("title").alias("category"))
ca_id = ca_id.withColumn("index", monotonically_increasing_id())
ca_title = ca_title.withColumn("index", monotonically_increasing_id())
df2 = ca_title.join(ca_id, "index", "outer").drop("index")
ca=ca.withColumnRenamed('category_id', 'id')
ca=ca.join(df2,'id',how='left').drop('id')

# COMMAND ----------

#combine two datasets into one
df=us.union(ca)
#Check missing values
from pyspark.sql.functions import count
def my_count(df):
  df.agg(*[count(c).alias(c) for c in df.columns]).show()

my_count(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Combine two datasets into one set and adjust the incorrect data type and Check missing values

# COMMAND ----------

#drop rows contains missing values
df=df.na.drop()
my_count(df)

# COMMAND ----------

# MAGIC %md
# MAGIC When importing the data, all variables were treated as string. There are some numerical variables, therefore we need to convert it into double datatype.

# COMMAND ----------

#transfer data type
from pyspark.sql.functions import col , column, to_date, unix_timestamp
num_col=['views','likes','dislikes','comment_count']
for n in num_col:
  df = df.withColumn(n, col(n).cast("double"))

# COMMAND ----------

# MAGIC %md
# MAGIC Add two new variables for future use. One is the ratio of likes over views which can indicates the percentage of likes, one is the ratio of dislikes over views indicating the percentage of dislikes.

# COMMAND ----------

import pyspark.sql.functions as func
df=df.withColumn('likes_ratio',func.round(df.likes/df.views,4))
df=df.withColumn('dislikes_ratio',func.round(df.dislikes/df.views,4))
df.printSchema

# COMMAND ----------

# MAGIC %md
# MAGIC Since the filtered dataset is not quite large, at the later part, it would be transfered into Pandas for the convenience of data visualizaiton.

# COMMAND ----------

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

#transfer into pandas.dataframe
pd_df=df.toPandas() 
pd_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC * Considering 'trending_date' and 'publish_time' are datetime variables, it is better to transfer them into datetime type instead of object type. 
# MAGIC * Remove some duplicates records by only keeping the lastest record

# COMMAND ----------

#transfer the time datetype
pd_df['trending_date'] = pd.to_datetime(pd_df['trending_date'], format='%y.%d.%m').dt.date
publish_time = pd.to_datetime(pd_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
pd_df['publish_time']=publish_time
#remove duplicates and only keep the lastest record
pd_df=pd_df.sort_values('publish_time',ascending=True).drop_duplicates(subset=['video_id'],keep='last')

# COMMAND ----------

#seperate the publish time into date, time and hour for future use
pd_df['publish_date'] = publish_time.dt.date
pd_df['publish_time'] = publish_time.dt.time
pd_df['publish_hour'] = publish_time.dt.hour

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Visualazation
# MAGIC #### 1. Correlations between views,likes,dislikes,comment_count,likes_ratio,dislikes_ratio.

# COMMAND ----------

corr=pd_df.loc[:,["views","likes","dislikes","comment_count",'likes_ratio','dislikes_ratio']].corr()
ax=sns.heatmap(corr,  vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=100),square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.title('Heatmap of numeric variables correlations ')
display(ax)

# COMMAND ----------

# MAGIC %md
# MAGIC From the correlation plot, we can see all numerical variables are either positive correlated or uncorrelated. 
# MAGIC 
# MAGIC - 'Dislikes' and 'likes' do not have a significant relation. However 'likes' and 'views' have a strong relation indicating that videos with more views tends to have more likes. But videos with more views does not tend to have more dislikes as well.
# MAGIC - 'likes_ratio' and 'dislikes_ratio' seem uncorrelated with other variables indicating these two variables are quite useful.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Which category is always trending?

# COMMAND ----------

#category plot
plt.close('all')
fig, ax = plt.subplots()
cat_df_us = pd_df['category'].value_counts().reset_index()
plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax=sns.barplot(x='category',y='index',data=cat_df_us)
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Catogories of trend videos")
display(ax)

# COMMAND ----------

# MAGIC %md
# MAGIC From the barplot,
# MAGIC * **Top**: Among all the trending videos, the number of Entertainment videos is significantly much higher than the other categories. It is over two times as the second highest category which is News & Politics.  
# MAGIC  + _*This phenomenon aligns with common sense, becuase people watch YouTube for entertainment when they are free in general. News & Politics videos often get trending because there are some breaking news that everyone cares or affect everyone in the area.*_
# MAGIC 
# MAGIC * **Bottom**: There are few trending videos that belong to Nonprofits & Activism and Movies category. In the plot, one cannot even see it.  
# MAGIC  + _*People do not really watch movies on YouTube and movies are always charged. Besides, movies are too long, no one would click on a three-hour movie even though it is trending. In additions, Nonprofits & Activism is a relatively small genre and thus few trending videos in this category.*_

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. When is the best time in a day to publish video?

# COMMAND ----------

plt.close('all')
fig, ax = plt.subplots()
plt.figure(figsize=(15,10))
ax=sns.countplot(x='publish_hour',data=pd_df,palette='Blues_r', order=pd_df['publish_hour'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Publish Hour')
plt.ylabel('Number of Videos')
plt.title('Publish Hour of trend videos')
display(ax)

# COMMAND ----------

# MAGIC %md
# MAGIC * In general, most of the trending videos are published in the afternoon or at night
# MAGIC * 15-17 pm seems to be the best time to release a video

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Top 10 most liked videos
# MAGIC _Found the top 10 videos that have the highest like-ratio_

# COMMAND ----------

from IPython.display import HTML
col=['thumbnail_link','title','views','likes_ratio','dislikes_ratio','category']
most_frequent=pd_df[col].sort_values("likes_ratio", ascending=False).head(10)

# Construction of HTML table with miniature photos assigned to the most popular movies
table_content = ''
max_title_length = 50

for date, row in most_frequent.T.iteritems():
    HTML_row = '<tr>'
    HTML_row += '<td><img src="' + str(row[0]) + '"style="width:100px;height:100px;"></td>'
    HTML_row += '<td>' + str(row[1]) + '</td>'
    HTML_row += '<td>' + str(row[2])  + '</td>'
    HTML_row += '<td>' + str(row[3]) + '</td>'
    HTML_row += '<td>' + str(row[4]) + '</td>'
    HTML_row += '<td>' + str(row[5]) + '</td>'
    
    table_content += HTML_row + '</tr>'

displayHTML(
  '<table><tr><th>Thumbnail</th><th>Title</th><th style="width:250px;">Views</th><th>Likes_ratio</th><th>Dislikes_ratio</th><th>Category</th></tr>{}</table>'.format(table_content))

# COMMAND ----------

# MAGIC %md
# MAGIC * Trending videos do not always have the most views. Many from the top 10 which have high like-ratio over views got into trending have only a few thousands views. 
# MAGIC * Likes_ratio seems to be an important part of YouTube trending videos Algorithm.
# MAGIC * Most liked videos do not domain in one or two categories, their categories vary.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Top 10 most viewd videos
# MAGIC _Found the top 10 videos that have the highest views_

# COMMAND ----------

col=['thumbnail_link','title','views','likes_ratio','dislikes_ratio','category']
pd_df[pd_df.category!='Music']
most_frequent=pd_df[col].sort_values("views", ascending=False).head(10)

# Construction of HTML table with miniature photos assigned to the most popular movies
table_content = ''
max_title_length = 50

for date, row in most_frequent.T.iteritems():
    HTML_row = '<tr>'
    HTML_row += '<td><img src="' + str(row[0]) + '"style="width:100px;height:100px;"></td>'
    HTML_row += '<td>' + str(row[1]) + '</td>'
    HTML_row += '<td>' + str(row[2])  + '</td>'
    HTML_row += '<td>' + str(row[3]) + '</td>'
    HTML_row += '<td>' + str(row[4]) + '</td>'
    HTML_row += '<td>' + str(row[5]) + '</td>'
    
    table_content += HTML_row + '</tr>'

displayHTML(
  '<table><tr><th>Thumbnail</th><th>Title</th><th style="width:250px;">Views</th><th>Likes_ratio</th><th>Dislikes_ratio</th><th>Category</th></tr>{}</table>'.format(table_content))

# COMMAND ----------

# MAGIC %md
# MAGIC * Most of the most viewed trending videos are music vedios
# MAGIC * When the videos get millions of views, the like ratio is not referable anymore
# MAGIC * Videos from popular singers or popular movie trailer can attract most of the users to click on and tend to get trending 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. What insight does TAGS show

# COMMAND ----------

#wordcloud
plt.close('all')
from wordcloud import WordCloud 
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


#tags_word = pd_df[pd_df['category']=='News & Politics']['tags'].str.lower().str.cat(sep=' ')
tags_word = pd_df['tags'].str.lower().str.cat(sep=' ')
tags_word = re.sub('[^A-Za-z]+', ' ', tags_word)
word_tokens = word_tokenize(tags_word)
en_stopwords= set(stopwords.words('english'))
filtered_sentence = [w for w in word_tokens if not w in en_stopwords]
without_single_chr = [word for word in filtered_sentence if len(word) > 2]
cleaned_data_title = [word for word in without_single_chr if not word.isdigit()]

from collections import Counter
counts=Counter(cleaned_data_title)

wordcloud = WordCloud(width=900,height=500, max_words=3000,relative_scaling=1).generate_from_frequencies(counts)

plt.figure(figsize=(14, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
display()

# COMMAND ----------

# MAGIC %md
# MAGIC The size of the word in the wordcloud is based on the frequency of a word in the tags. The higher frequency, the bigger the word size is.  
# MAGIC <br>
# MAGIC 
# MAGIC  * 'News' and 'Trump' apprently happen very frequently in the news&politics category videos and news&politics do have the second highest number among all the trending videos.
# MAGIC  * Same for the music category with the 'songs' and 'music'
# MAGIC  * Positive words such as 'funny', 'comedy', 'family', 'best', 'challenge', 'highlights' are quite frequent among trending videos.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recomendations to YouTubers  
# MAGIC <br>
# MAGIC - **Do ask your viewers to give likes to the video** becuase it can still push your video to be trending regardless of how many views your video have
# MAGIC - **Publish your video during 15-17 pm of your day** because most of the trending videos are released during that time
# MAGIC - **Tag your video with positive words** because the most frequent words are either neutural function words or positive words
