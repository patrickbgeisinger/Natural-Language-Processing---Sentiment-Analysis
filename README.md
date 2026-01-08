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
```

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
- **Reviews analyzed**: 28332
- **Date range**: May 2019

### Testing sentiment on sample reviews:
Review: I order 3 of them and one of the item is bad quality. Is missing backup spring so I have to put a pcs of aluminum to make the battery work.
Sentiment: Negative - Polarity Score: -0.4500

Review: Bulk is always the less expensive way to go for products like these
Sentiment: Negative - Polarity Score: -0.3333

Review: Well they are not Duracell but for the price i am happy.
Sentiment: Positive - Polarity Score: 0.8000

Review: Seem to work as well as name brand batteries at a much better price
Sentiment: Positive - Polarity Score: 0.5000

Review: These batteries are very long lasting the price is great.
Sentiment: Positive - Polarity Score: 0.2450

### Sentiment Distribution
- Positive reviews: 90.16%
- Negative reviews: 9.84%
- Neutral reviews: 4.26%

## Key Insights

### Model Strengths
- Effective at identifying clear positive and negative sentiment
- Fast processing speed for large datasets
- Good handling of common review patterns
- Reliable polarity scoring

### Model Limitations
- May struggle with sarcasm and irony
- Context-dependent meanings can be challenging
- Mixed sentiment reviews may be misclassified
- Domain-specific terminology requires fine-tuning

## 📖 Documentation
For detailed information about the analysis process, preprocessing steps, and evaluation metrics, please refer to:
- **sentiment_analysis_report.pdf** - Comprehensive project report

## 👤 Author
Patrick Geisinger

## 📄 License
This project is part of the Arizona State University bootcamp curriculum.

## 🙏 Acknowledgments
- Arizona State University for the project guidelines
- Datafiniti for the Amazon reviews dataset
- spaCy and TextBlob development teams

---

**Note**: This is an educational project for learning NLP and sentiment analysis techniques.
