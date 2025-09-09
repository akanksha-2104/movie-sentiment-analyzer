🎬 Movie Review Sentiment Analyzer

A machine learning web application that analyzes the sentiment of movie reviews using Natural Language Processing (NLP) and Logistic Regression.

📖 How I Built It
1. Text Preprocessing

Converted all text to lowercase

Removed HTML tags, punctuation, and special characters

Removed stopwords using NLTK

Applied lemmatization to normalize words

2. Feature Extraction

Used TF-IDF Vectorization with a vocabulary of up to 5000 features

Applied both unigram and bigram analysis to capture context

3. Model Training

Implemented Logistic Regression as the classifier

Trained the model on the IMDB Movie Reviews dataset

Achieved ~88% accuracy in classifying reviews as positive or negative

4. Web Application

Built an interactive interface with Streamlit

Added input box for user reviews and displayed sentiment prediction with confidence scores

Integrated sample reviews for quick testing

5. Deployment

Packaged all dependencies in requirements.txt

Deployed the application to Streamlit Cloud for easy public access

🛠️ Tech Stack

Python: Core programming language

Scikit-learn: Machine learning and Logistic Regression

NLTK: Text preprocessing (stopwords, lemmatization)

Streamlit: Web application framework

Pandas & NumPy: Data handling and numerical operations
