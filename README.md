# Natural-Language-Processing---Sentiment-Analysis
This project performs sentiment analysis on Amazon product reviews using Natural Language Processing (NLP) techniques.

## Project Overview
This project performs sentiment analysis on Amazon product reviews using Natural Language Processing (NLP) techniques. The system analyzes customer reviews to determine whether they express positive, negative, or neutral sentiment, helping businesses understand customer opinions and feedback at scale.

## 🚀 Features
- Text preprocessing (stop word removal, text cleaning)
- Sentiment classification (positive, negative, neutral)
- Polarity score calculation (-1 to +1 scale)
- Review similarity comparison
- Sample testing on product reviews
- Comprehensive analysis report

## 💻 Technologies Used
- **Python 3.8+**
- **spaCy** - NLP library for text processing
- **TextBlob** - Sentiment analysis
- **pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Jupyter Notebook** - Interactive development

## Usage
### Install required packages
```bash
pip install spacy pandas numpy textblob spacytextblob jupyter
```

### Download spaCy language model
```bash
python -m spacy download en_core_web_md
```

### Download TextBlob corpora
```bash
python -m textblob.download_corpora
```

### Running the Jupyter Notebook
```bash
jupyter notebook sentiment_analysis.ipynb
```
### Using the Sentiment Analysis Function
```python
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the model
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacytextblob')

# Analyze sentiment
def predict_sentiment(review_text):
    doc = nlp(review_text)
    polarity = doc._.blob.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Test it
review = "This product is amazing! I love it."
print(predict_sentiment(review))

## Project Workflow

### 1. Data Loading
- Load Amazon product reviews dataset from Kaggle
- Dataset: Datafiniti Amazon Consumer Reviews (May 2019)
- Extract the 'reviews.text' column for analysis

### 2. Data Preprocessing
- Remove missing values using `dropna()`
- Remove stop words (common words like "the", "is", "of")
- Clean text using `lower()`, `strip()`, and `str()` methods
- Tokenize reviews for processing

### 3. Sentiment Analysis Model
- Implement using spaCy's `en_core_web_md` model
- Integrate TextBlob for sentiment scoring
- Calculate polarity scores (-1 to +1 range)
- Classify reviews as positive, negative, or neutral

### 4. Model Testing
- Test on sample product reviews
- Evaluate accuracy and performance
- Compare review similarities using `similarity()` function

### 5. Results & Reporting
- Document preprocessing steps
- Analyze model strengths and limitations
- Provide insights and recommendations

## Results

### Dataset Information
- **Source**: Datafiniti Amazon Consumer Reviews (Kaggle)
- **Reviews analyzed**: [Add your number]
- **Date range**: May 2019
