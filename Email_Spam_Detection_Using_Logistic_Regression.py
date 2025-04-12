#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #data manipulation


# In[2]:


import re #Regular expression for calculations and pattern making


# In[3]:


import nltk #Part of NLP


# In[4]:


from nltk.corpus import stopwords


# In[5]:


from nltk.stem import PorterStemmer


# In[6]:


nltk.download("stopwords")


# In[7]:


stemmer = PorterStemmer()


# In[8]:


stop_words=set(stopwords.words("english"))


# In[23]:


df = pd.read_csv("Email.csv",encoding="latin-1")[["text","spam"]]


# In[24]:


df.head()


# In[25]:


df.columns = ["message","label"] #rename the columns


# In[26]:


df.head()


# In[27]:


def preprocess_text(text):
    text = re.sub(r"\W"," ",text) #remove Special Character
    text = text.lower() #convert into Lowercase
    words = text.split()
    words= [stemmer.stem(word) for word in words if word not in stop_words]
    # Remove the Stop Words and Stem Words
    return " ".join(words)


# In[28]:


df["cleaned_message"]= df["message"].apply(preprocess_text)


# In[29]:


df.head()


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer #converting text into Numerical


# In[31]:


from sklearn.model_selection import train_test_split #distributing data into train and test


# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


from sklearn.metrics import accuracy_score,classification_report


# In[34]:


vectorizer = TfidfVectorizer(max_features = 3000)
X = vectorizer.fit_transform(df["cleaned_message"]) #input data


# In[35]:


y = df["label"] #output data


# In[36]:


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8, random_state =42)


# In[37]:


model = LogisticRegression()


# In[38]:


model.fit(X_train,y_train) # using train data we can predict


# In[39]:


y_pred = model.predict(X_test)


# In[40]:


print(f"accuracy: {accuracy_score(y_test,y_pred) * 100:.2f}%")


# In[41]:


print(classification_report(y_test,y_pred))


# In[42]:


def predict_email(email_text):
    processed_data = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_data])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0]==1 else "Ham - Not Spam"


# In[43]:


email = '''Hello Bhavesh,

Exciting news: Fliki has been nominated for the Product Hunt Golden Kitty Awards 2024 in the “AI for Video” category!

This is a huge milestone for our team, and it wouldn’t have been possible without your continued support. Now, we need your help to bring home the win.


If Fliki has helped you create awesome videos, or if you simply believe in what we’re building, please take a moment to cast your vote:

Vote for Fliki
Every vote counts—and your support truly means the world to us. 

Thank you for being part of this journey!

Note: Voting ends in 12 hours '''


# In[44]:


print(f"Email: {email}\n Prediction :{predict_email(email)}")


# In[45]:


email_2 = """Tired of your 9-to-5 job? Ready to make BIG MONEY from the comfort of your home?

We are offering a revolutionary online work opportunity — no experience needed!  
Start earning **$10,000 or more per week**!

✅ Flexible hours  
✅ Instant payouts  
✅ Limited spots available!

👉 Sign up today: [Start Earning Now](http://fastmoney-onlinejobs.com)

Don't miss this golden chance to change your life forever!

Sincerely,  
Global Online Careers Team

 """


# In[47]:


print(f"Email: {email_2}\n Prediction :{predict_email(email_2)}")


# In[ ]:




