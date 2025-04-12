# Spam Email Classifier using Logistic Regression

This project implements a **Spam Email Classifier** using **Logistic Regression**.  
The model classifies email texts as **Spam** or **Not Spam**.

## Key Features:
- **Logistic Regression** algorithm for spam classification.
- **Text Preprocessing** using **TfidfVectorizer**.
- Trained and tested on an email dataset.
- Built to run on **Google Colab** with minimal dependencies.

## Libraries Used:
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For machine learning model implementation (Logistic Regression) and text vectorization (TfidfVectorizer).
- **NLTK**: For text cleaning (stopwords removal and stemming).

## Workflow:
1. **Data Preprocessing**: 
    - Load the email dataset.
    - Clean the text (removing special characters, converting to lowercase).
    - Remove stopwords and perform stemming.
2. **Model Building**:
    - Use **Logistic Regression** to train the classifier.
    - Evaluate model performance using **accuracy** and **classification report**.
3. **Prediction**:
    - Input an email and get a prediction of whether it's **Spam** or **Not Spam**.

## How to Use:
1. Upload your email dataset or use the provided sample data.
2. Run all cells in sequence in **Google Colab**.
3. Use the provided `predict_email` function to classify any custom email text.

## Example Email:
```text
Hello Bhavesh,
Exciting news: Fliki has been nominated for the Product Hunt Golden Kitty Awards 2024 in the “AI for Video” category!
This is a huge milestone for our team, and it wouldn’t have been possible without your continued support. Now, we need your help to bring home the win.
If Fliki has helped you create awesome videos, or if you simply believe in what we’re building, please take a moment to cast your vote:
Vote for Fliki
Every vote counts—and your support truly means the world to us.
Thank you for being part of this journey!
Note: Voting ends in 12 hours
