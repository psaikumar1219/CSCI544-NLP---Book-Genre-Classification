#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')


# # 1. DATA PREPARATION

# ## Reading data into a dataframe. We use the pandas read_table method and give the dataset name. We also use the property "on_bad_lines = skip" which does not take the rows which probably might not follow the same format as others and not be favorable for the further steps.

# In[2]:


fd = pd.read_table('amazon_reviews_us_Beauty_v1_00.tsv', on_bad_lines='skip')


# In[3]:


fd


# ## Taking only two columns. We only take the columns "star_rating" and "review_body" which we use for the further processing. We simply do not consider other columns and hence do not include in this dataframe.

# In[4]:


df2 = fd[["star_rating","review_body"]]


# In[5]:


df2


# In[6]:


c1 = df2[df2['star_rating'] == '1']


# In[7]:


c2 = df2[df2['star_rating'] == '2']


# In[8]:


frames = [c1,c2]


# In[9]:


class1 = pd.concat(frames)


# ## We only take the random 20000 rows from the data having ratings 1 and 2. We use sample() to choose the random 20000 rows from the dataframe.

# In[10]:


cls1 = class1.sample(20000)


# In[11]:


c3 = df2[df2['star_rating'] == '3']


# ## We only take the random 20000 rows from the data having rating 3. We use sample() to choose the random 20000 rows from the dataframe.

# In[12]:


cls2 = c3.sample(20000)


# In[13]:


c4 = df2[df2['star_rating'] == '4']


# In[14]:


c5 = df2[df2['star_rating'] == '5']


# In[15]:


frames1 = [c4,c5]


# In[16]:


class3 = pd.concat(frames1)


# ## We only take the random 20000 rows from the data having ratings 4 and 5. We use sample() to choose the random 20000 rows from the dataframe.

# In[17]:


cls3 = class3.sample(20000)


# In[18]:


frames2 = [cls1,cls2,cls3]


# ## We take all the dataframes with 20000 rows having each class into a new dataframe with 60000 rows.

# In[19]:


samp = pd.concat(frames2)


# In[20]:


samp


# # 2. DATA CLEANING

# ## Average length of the reviews BEFORE cleaning.

# In[21]:


len_before_DC = samp['review_body'].str.len().mean()


# ## Removing the URLs from the reviews. Used the str.replace() method with the regular expression which captures every matching string and replaces it with an empty space.
# 

# In[22]:


samp['review_body'] = samp['review_body'].str.replace('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', ' ')


# ## Removing HTML tags from the reviews. Uses str.replace() method with the regular expression which captures every HTML tag and replaces it with an empty space. 
# 

# In[23]:


samp['review_body'] = samp['review_body'].str.replace(r'<[^<>]*>', '', regex=True)


# ## Removing Contractions (expanding the words)

# ### Installing the contractions.

# In[24]:

# ### Function to remove contractions in every review. First we check the type of each review and iff it is a string then we replace it with a string without contractions by using contractions.fix().
# 

# In[25]:


import contractions
def cont_to_exp(x):
    if type(x) is str:
        x = x.replace(x, contractions.fix(x))
        return x
    else:
        return x


# ### We use lambda to apply the function "cont_to_exp()" to every review. 

# In[26]:


samp['review_body'] = samp['review_body'].apply(lambda x : cont_to_exp(x))


# ## Converting every character into its corresponding lower case character.

# In[27]:


samp = samp.apply(lambda x: x.astype(str).str.lower())


# ## Removing every non-alphabetic character.

# In[28]:


samp['review_body'] = samp['review_body'].str.replace('[^a-z]', ' ')


# ## Removing extra spaces.

# In[29]:


samp['review_body'] = samp['review_body'].str.replace('  ', ' ')


# ## Average length of the reviews AFTER cleaning.

# In[30]:


len_after_DC = samp['review_body'].str.len().mean()


# In[31]:

# In[32]:


print(len_before_DC, ',', len_after_DC)


# In[33]:


nosamp = samp


# # 3. DATA PREPROCESSING

# ## Removing stop words. 

# ### We store all the stop words in "stops". The function remove_stops() goes through every word in the string and if a stop word is found it is replaced with an empty space. Then lambda is used to apply remove_stops() to every string in the reviews column, thereby removing the stop words from every review.

# In[34]:


import nltk
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

def remove_stops(x):
    return " ".join([word for word in str(x).split() if word not in stops])

samp['review_body'] = samp['review_body'].apply(lambda x : remove_stops(x))


# ## Lemmatization

# ### We import the WordNetLemmatizer and we create an instance of the WordNetLemmatizer(). We use lemmatize() for each word in the string in the reviews by using the function lemma_data() and store the list of lemmatized words in the new column "lemmatized". We apply this function to every string in the review_body column using lambda.

# In[35]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemma_data(x):
    text = [lemmatizer.lemmatize(word) for word in str(x).split()]
    return text

samp['lemmatized'] = samp['review_body'].apply(lambda x : lemma_data(x))


# ### Converting the list in lemmatized column to a string in original reviews column and delete the lemmatized column created in the previous step.

# In[36]:


samp['review_body'] = [' '.join(map(str, l)) for l in samp['lemmatized']]


# In[37]:


del samp['lemmatized']


# ## Length of the reviews AFTER data preprocessing.

# In[38]:


len_after_DP = samp['review_body'].str.len().mean()


# In[39]:

# In[40]:


print(len_after_DC, ',', len_after_DP)


# # Creating a new column "classification". We populate the classification column with class labels which the functions returns for a rating.
# 

# In[41]:


def class_category(row):
    if row['star_rating'] == '1' or row['star_rating'] == '2':
        val = 1
    elif row['star_rating'] == '3':
        val = 2
    else:
        val = 3
    return val

samp['classification'] = samp.apply(class_category, axis=1)


# # 4. FEATURE EXTRACTION

# ## Here we import the tfidfVectorizer to convert the reviews into a matrix of TFIDF features. The ngram_range would give us the range of n_grams to be included in the Bag of Words. (1,3) would give us n_grams from one to three words. X is the matrix of TFIDF features from the reviews.
# 

# In[42]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,3))

X = vectorizer.fit_transform(samp['review_body'])


# # Splitting the dataset into test and train data.

# ## We import the train_test_split to perform the splitting of the dataset. We split the feature vector X and respective classifications into x_train, x_test, y_train, y_test respectively. The test_size = 0.2 refers to the percentage size of the test data of the whole dataset. It helps us split the dataset into training data and test data in a 80-20% fashion, with training data at 80% and testing data at 20%. The stratify parameter will help us split the data such that every split is having same number of members from each class, which might be better for the training and overall performance of the model. Here, we set the stratify to the classification column.

# In[43]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,samp['classification'],test_size = 0.2,stratify=samp['classification'])


# # Importing the important metrics from the sklearn metrics such as accuracy_score, precision_recall_fscore_support, and classification_report. 

# In[44]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


# # printMetrics() function prints the precision, recall, f1-score, and their respective averages for each class label. It takes the testing sample of the class labels and output of the predict() method in each model as parameters, and prints the precision, recall, f1-score separated by comma. It gives the average of the above metrics in the last line. First we get a classification report in the dictionary form and we store its transpose in a dataframe. We iterate through the dictionary and take only the metrics we need.

# In[45]:


def printMetrics(y_test, label):
    cr = classification_report(y_test, label, output_dict=True)
    report = pd.DataFrame(cr).transpose()
    for i in range(4):
        if i==3:
            print(f'{report.iloc[i+1]["precision"]}, {report.iloc[i+1]["recall"]}, {report.iloc[i+1]["f1-score"]}\n')
        else:
            print(f'{report.iloc[i]["precision"]}, {report.iloc[i]["recall"]}, {report.iloc[i]["f1-score"]}\n')
        

# # -------------------------------------------------------------------------------------------

# # Without Data Preprocessing

# ## I observed that the metrics for the data which is not preprocessed are high compared to the preprocessed data. I did not remove the stop words and did not perform lemmatization (which is an optional step in the data preprocessing), and did other step just as above, and I ended up getting high precision, recall, f1-score, and accuracy. 

# ## 1. DATA PREPARATION

# ### Took the same data as above prior to the data cleaning step.

# ## 2. DATA CLEANING

# ### Took the same data as above prior to the data preprocessing step.

# ## 3. DATA PREPROCESSING - **SKIPPED**

# ## 4. FEATURE EXTRACTION

# In[54]:


nX = vectorizer.fit_transform(nosamp['review_body'])


# ## Training and testing data split in similar fashion as above.

# In[55]:


nx_train, nx_test, ny_train, ny_test = train_test_split(nX,nosamp['classification'],test_size = 0.2,stratify=nosamp['classification'])


# ## 5. PERCEPTRON

# In[56]:


from sklearn.linear_model import Perceptron

nmodel = Perceptron()
nmodel.fit(nx_train, ny_train)
nlabelPredict = nmodel.predict(nx_test)
accuracy_score(ny_test, nlabelPredict)

printMetrics(ny_test, nlabelPredict) 


# ## Accuracy for Perceptron.

# In[57]:


accuracy_score(ny_test, nlabelPredict)


# ## 6. SVM

# In[58]:


from sklearn.svm import LinearSVC

nSVMmodel = LinearSVC()
nSVMmodel.fit(nx_train, ny_train)
nSVMLabelPredict = nSVMmodel.predict(nx_test)
accuracy_score(ny_test, nSVMLabelPredict)

printMetrics(ny_test, nSVMLabelPredict) 


# ## Accuracy for SVM.

# In[59]:


accuracy_score(ny_test, nSVMLabelPredict)


# ## 7. LOGISTIC REGRESSION

# In[60]:


from sklearn.linear_model import LogisticRegression

nLRmodel = LogisticRegression()
nLRmodel.fit(nx_train, ny_train)
nLRLabelPredict = nLRmodel.predict(nx_test)
accuracy_score(ny_test, nLRLabelPredict)

printMetrics(ny_test, nLRLabelPredict) 


# ## Accuracy for Logistic Regression.

# In[61]:


accuracy_score(ny_test, nLRLabelPredict)


# ## 8. MULTINOMIAL NAIVE BAYES

# In[62]:


from sklearn.naive_bayes import MultinomialNB

nNBmodel = MultinomialNB()
nNBmodel.fit(nx_train, ny_train)
nNBLabelPredict = nNBmodel.predict(nx_test)
accuracy_score(ny_test, nNBLabelPredict)

printMetrics(ny_test, nNBLabelPredict) 


# ## Accuracy for Multinomial Naive Bayes.

# In[63]:


accuracy_score(ny_test, nNBLabelPredict)


