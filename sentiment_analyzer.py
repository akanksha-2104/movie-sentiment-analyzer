# sentiment_analyzer.py
# Main application file - this is your complete web app

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Download NLTK data (runs once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        with st.spinner("Downloading language data..."):
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text for better ML performance"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords and lemmatize
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words 
                if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    except:
        # Fallback if NLTK data not available
        words = text.split()
        words = [word for word in words if len(word) > 2]
        return ' '.join(words)

# Load sample data
@st.cache_data
def load_sample_data():
    """Create sample dataset for demonstration"""
    # Positive reviews
    positive_reviews = [
        "This movie was absolutely fantastic! Great acting and brilliant storyline.",
        "Amazing cinematography and outstanding performances by all actors.",
        "One of the best films I've ever seen! Highly recommend this masterpiece.",
        "Incredible movie with great special effects and superb acting.",
        "Excellent direction and wonderful cast. A must-watch film.",
        "Loved every minute of it! Engaging plot and fantastic characters.",
        "Brilliant storytelling with excellent character development.",
        "Spectacular visuals and amazing soundtrack. Truly impressive.",
        "Outstanding movie with great dialogue and perfect pacing.",
        "Remarkable film with incredible performances and beautiful scenes.",
        "Fantastic movie! The plot was engaging and the acting was superb.",
        "Absolutely loved this film. Great story and amazing visuals.",
        "Perfect movie with excellent direction and outstanding cast.",
        "This is cinema at its finest. Brilliant and captivating throughout.",
        "Wonderful film with great emotions and powerful performances.",
        "Exceptional movie that kept me entertained from start to finish.",
        "Masterpiece of filmmaking with incredible attention to detail.",
        "Loved the characters and the beautiful storytelling in this film.",
        "Outstanding performances and a gripping storyline throughout.",
        "This movie exceeded all my expectations. Truly remarkable work."
    ]
    
    # Negative reviews
    negative_reviews = [
        "Terrible movie, complete waste of time. Poor acting and boring plot.",
        "One of the worst movies I've ever seen. Completely disappointing.",
        "Not worth watching. Very predictable and poorly executed.",
        "Boring and confusing. Could not even finish watching it.",
        "Poor dialogue and weak character development. Very disappointing.",
        "Awful movie with terrible acting and nonsensical plot.",
        "Extremely boring with no redeeming qualities whatsoever.",
        "Poorly made film with bad direction and worse acting.",
        "Disappointing and frustrating. Complete waste of money.",
        "Terrible script and unconvincing performances throughout.",
        "This movie was painful to watch. Terrible in every aspect.",
        "Poorly written with bad acting and no coherent storyline.",
        "Waste of time. Boring plot and terrible character development.",
        "Disappointing movie with poor direction and weak performances.",
        "Bad film with confusing plot and unconvincing acting.",
        "Terrible movie that failed to engage on any level.",
        "Poor quality film with bad writing and worse execution.",
        "Completely boring movie with no redeeming features at all.",
        "Awful film with terrible dialogue and poor character development.",
        "This movie was a complete disaster. Poorly made in every way."
    ]
    
    # Create DataFrame
    reviews = positive_reviews + negative_reviews
    sentiments = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    
    return df

# Train model
@st.cache_resource
def train_sentiment_model():
    """Train the sentiment analysis model"""
    # Download NLTK data
    download_nltk_data()
    
    # Load data
    df = load_sample_data()
    
    # Preprocess reviews
    with st.spinner("Processing text data..."):
        df['processed_review'] = df['review'].apply(preprocess_text)
    
    # Remove empty reviews
    df = df[df['processed_review'].str.len() > 0]
    
    # Prepare features and target
    X = df['processed_review']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, accuracy

# Predict sentiment
def predict_sentiment(text, model, vectorizer):
    """Predict sentiment of input text"""
    if not text.strip():
        return None, None
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        if not processed_text:
            return None, None
        
        # Vectorize
        vectorized_text = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]
        
        return prediction, probability
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

# Main application
def main():
    # Title and description
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("""
    ### Analyze the sentiment of movie reviews using Machine Learning
    """)
    
    # Sidebar
    st.sidebar.header("ℹ️ About This App")
    st.sidebar.info(
        "**How it works:**\n\n"
        "1. **Text Preprocessing**: Cleans and prepares the review text\n"
        "2. **Feature Extraction**: Converts text to numerical features using TF-IDF\n"
        "3. **Classification**: Uses Logistic Regression to predict sentiment\n"
        "4. **Results**: Shows prediction with confidence score"
    )
    
    # Load model
    with st.spinner("🤖 Loading AI model..."):
        model, vectorizer, accuracy = train_sentiment_model()
    
    st.sidebar.success(f"✅ Model loaded successfully!")
    st.sidebar.metric("Model Accuracy", f"{accuracy:.1%}")
    
    # Main interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📝 Enter Your Movie Review")
        
        # Text input
        user_input = st.text_area(
            "Type or paste a movie review here:",
            height=150,
            placeholder="Example: This movie was absolutely amazing! Great acting and fantastic storyline that kept me engaged throughout..."
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if analyze_button:
            if user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    prediction, probability = predict_sentiment(user_input, model, vectorizer)
                
                if prediction is not None:
                    # Display prediction
                    if prediction == 1:
                        st.success("😊 **Positive Sentiment**")
                        confidence = probability[1] * 100
                        st.balloons()
                    else:
                        st.error("😞 **Negative Sentiment**")
                        confidence = probability[0] * 100
                    
                    # Show confidence
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                    
                    # Progress bar for confidence
                    st.progress(confidence / 100)
                    
                    # Probability breakdown
                    st.subheader("🎯 Detailed Probabilities")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        st.metric("Negative", f"{probability[0]:.1%}")
                    
                    with prob_col2:
                        st.metric("Positive", f"{probability[1]:.1%}")
                    
                    # Interpretation
                    if confidence > 80:
                        confidence_text = "Very confident"
                    elif confidence > 60:
                        confidence_text = "Moderately confident"
                    else:
                        confidence_text = "Low confidence"
                    
                    st.info(f"**Interpretation**: {confidence_text} in this prediction")
                
                else:
                    st.warning("❌ Could not analyze this text. Please try a different review.")
            else:
                st.warning("⚠️ Please enter a movie review to analyze!")
        
        else:
            st.info("👆 Enter a review and click 'Analyze Sentiment' to see results")
    
    # Divider
    st.divider()
    
    # Sample reviews section
    st.subheader("🎬 Try These Sample Reviews")
    st.markdown("Click on any sample review to test the analyzer:")
    
    sample_reviews = [
        {
            "text": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout the entire film.",
            "label": "Sample Positive Review"
        },
        {
            "text": "Terrible movie, complete waste of time. Poor acting and a boring, predictable storyline that put me to sleep.",
            "label": "Sample Negative Review"
        },
        {
            "text": "One of the best films I've seen this year! Amazing cinematography, brilliant performances, and a captivating story.",
            "label": "Another Positive Review"
        },
        {
            "text": "Disappointing and confusing movie. Could not even finish watching it. Very poorly made with terrible dialogue.",
            "label": "Another Negative Review"
        }
    ]
    
    # Create sample review buttons
    cols = st.columns(2)
    for i, sample in enumerate(sample_reviews):
        with cols[i % 2]:
            if st.button(sample["label"], key=f"sample_{i}", use_container_width=True):
                # Analyze sample
                prediction, probability = predict_sentiment(sample["text"], model, vectorizer)
                
                if prediction is not None:
                    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
                    confidence = max(probability) * 100
                    
                    # Show sample results
                    st.markdown(f"**Review**: {sample['text']}")
                    st.markdown(f"**Predicted Sentiment**: {sentiment}")
                    st.markdown(f"**Confidence**: {confidence:.1f}%")
                    
                    # Add some spacing
                    st.markdown("")
    
    # Technical details section
    with st.expander("🔧 Technical Details & Model Information"):
        st.markdown("""
        ### 🧠 Machine Learning Model
        - **Algorithm**: Logistic Regression
        - **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Vocabulary Size**: 2000 features
        - **N-grams**: 1-2 (single words and word pairs)
        
        ### 📊 Text Preprocessing Steps
        1. **Lowercase conversion**: Normalize text case
        2. **HTML tag removal**: Clean web-scraped content
        3. **Special character removal**: Focus on words only
        4. **Stopword removal**: Remove common words like 'the', 'is', 'and'
        5. **Lemmatization**: Convert words to base form (e.g., 'running' → 'run')
        
        ### 🎯 Model Performance
        - **Training Data**: 40 sample movie reviews
        - **Accuracy**: {:.1%} on test data
        - **Classes**: Binary classification (Positive/Negative)
        
        ### 🚀 Future Enhancements
        - Expand to larger IMDB dataset (50,000+ reviews)
        - Implement deep learning models (LSTM, BERT)
        - Add aspect-based sentiment analysis
        - Include neutral sentiment classification
        """.format(accuracy))
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666666;'>
        <p>Built with ❤️ using Python, Streamlit, and scikit-learn</p>
        <p>📧 Contact: akankshaakawale@gmail.com | 🔗 GitHub: github.com/akanksha-2104</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
