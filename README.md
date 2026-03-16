# Applied NLP for Topic Modelling: Mining Customer Dissatisfaction Drivers at Scale

> **Bridging NLP theory and business intelligence** — using BERTopic, LDA, BERT emotion analysis, and LLM-based extraction to decode 30,000+ customer reviews into actionable operational insights for a major fitness operator.

---

## 📋 Project Overview

**Client Context:** PureGym Group — one of the world's largest value fitness operators with 2M+ members and 600+ gyms across the UK, Denmark, and Switzerland.

**Business Problem:** PureGym needed to move beyond manual review processing to systematically identify what drives negative customer sentiment across its network — and where.

**Approach:** Built an end-to-end NLP pipeline that ingests multi-platform review data (Google + Trustpilot), applies multiple topic modelling techniques, and surfaces location-specific, theme-specific dissatisfaction drivers with strategic recommendations.

---

## 🔬 Technical Pipeline

```
Raw Reviews (30,000+)
    │
    ├── Data Cleaning & Preprocessing (NLTK stopwords, tokenisation, normalisation)
    │
    ├── Exploratory Analysis (frequency distributions, word clouds, sentiment filtering)
    │
    ├── Topic Modelling
    │     ├── BERTopic → 44 distinct semantic topics from negative reviews
    │     └── Gensim LDA → 10-topic validation model
    │
    ├── Emotion Analysis
    │     └── BERT (bhadresh-savani/bert-base-uncased-emotion) → anger-filtered deep dive
    │
    └── LLM Exploration
          └── Falcon-7b-instruct → topic extraction & actionable suggestion generation
```
### Sample Codes

Using NLTK formost frequent words distribution and Word Cloud (Bag of Word) Pre-Processing.
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Define preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Join back into a string for FreqDist
    return ' '.join(tokens)

# Apply preprocessing to Google reviews
google_df['cleaned_comment'] = google_df['Comment'].apply(preprocess_text)

# Apply preprocessing to Trustpilot reviews
trustpilot_df['cleaned_review_content'] = trustpilot_df['Review Content'].apply(preprocess_text)

print("Text preprocessing complete. New 'cleaned_comment' and 'cleaned_review_content' columns created.")

from nltk.probability import FreqDist

# Get all words from Google reviews
google_words = ' '.join(google_df['cleaned_comment']).split()
google_freq_dist = FreqDist(google_words)

# Get all words from Trustpilot reviews
trustpilot_words = ' '.join(trustpilot_df['cleaned_review_content']).split()
trustpilot_freq_dist = FreqDist(trustpilot_words)

print("Frequency distribution for Google reviews (top 10):")
print(google_freq_dist.most_common(10))

print("\nFrequency distribution for Trustpilot reviews (top 10):")
print(trustpilot_freq_dist.most_common(10))

```
### Key Technical Highlights

| Technique | Tool / Model | Purpose |
|:---|:---|:---|
| **Topic Modelling** | BERTopic | Semantic clustering of negative reviews into 44 granular topics |
| **Topic Modelling** | Gensim LDA | Cross-validation of BERTopic findings; surfaced multilingual data quality issue |
| **Emotion Classification** | BERT (`bert-base-uncased-emotion`) | Filtered reviews by dominant emotion (anger) for targeted analysis |
| **Text Generation** | Falcon-7b-instruct (LLM) | Extracted top-3 topics per review; generated actionable recommendations |
| **Preprocessing** | NLTK, regex | Tokenisation, stopword removal, case normalisation |
| **Visualisation** | Word clouds, matplotlib, BERTopic built-in (intertopic distance, heatmaps, bar charts) | Thematic and geographic pattern identification |

---

## 📊 Key Findings

### Top 5 Negative Sentiment Drivers (BERTopic)

| Rank | Theme | Signal Words | Implication |
|:---|:---|:---|:---|
| 1 | **Hygiene** | toilets, cleaning, dirty, smell, changing | #1 detractor — changing room cleanliness |
| 2 | **Climate Control** | air, conditioning, hot, aircon | HVAC failures, especially pre-summer |
| 3 | **Access Issues** | pin, code, access | Digital entry system reliability |
| 4 | **Parking** | parking, fine, car park | Administrative friction |
| 5 | **Staff Conduct** | rude, manager | Isolated but high-impact negative interactions |

### Geographic Concentration
- **London locations** (Stratford, Enfield, Hayes, Swiss Cottage) are disproportionately represented in negative reviews across both platforms
- High footfall ≠ high negativity — specific failing locations drive the negative metrics, not volume

### Model Comparison: BERTopic vs. LDA
- ![BERTopic](/images/bertopics.png)
- ![BERTopic2](/images/bertopics2.png)
- **BERTopic** excelled at separating semantically similar but distinct issues (e.g., "Air Conditioning" vs. general "Facilities")

- - ![Gensim LDA](/images/LDAgensim.png)
- **LDA** surfaced a critical data quality insight: Danish stopwords in Topic 4 revealed unfiltered non-English reviews, flagging the need for multilingual preprocessing
- Both models converged on the same core themes, validating the findings

### LLM Exploration
- Falcon-7b-instruct faced reliability constraints in constrained compute environments, highlighting the current advantage of specialised topic modelling algorithms over generative LLMs for large-scale text processing

---

## 🎯 Strategic Recommendations

1. **Hygiene Intervention** — Audit cleaning schedules for changing rooms at high-complaint locations
2. **London Task Force** — Regional investigation into overcrowding and management at flagged gyms
3. **HVAC Maintenance** — Proactive air conditioning servicing before summer months
4. **Language Detection** — Implement multilingual routing in the feedback pipeline
5. **Digital Entry Review** — Improve PIN/app access reliability and onboarding clarity

---

## 🛠️ Tech Stack

```
Python | NLTK | BERTopic | Gensim | Hugging Face Transformers | BERT | Falcon-7b | 
Pandas | NumPy | Matplotlib | Seaborn | Scikit-Learn | TensorFlow/Keras | Google Colab
```

---

## 📁 Repository Structure

```
├── notebooks/
│   └── NLP_Topic_Modelling_PureGym.ipynb    # Full analysis notebook
├── reports/
│   └── NLP_Topic_Modelling_Report.pdf       # Executive summary report
├── data/                                     # Data directory (not included — proprietary)
└── README.md
```

---

## 📎 Context

This project was completed as part of the **University of Cambridge Data Science with Machine Learning** programme (Module C301: NLP & Topic Modelling). It demonstrates the practical application of multiple NLP techniques to a real-world business problem, moving beyond keyword frequency analysis to semantic understanding of customer sentiment at scale.

---

## 🤝 Connect

- **Portfolio:** [raffael.notion.site/portfolio](https://raffael.notion.site/portfolio)
- **LinkedIn:** [linkedin.com/in/raffaelyg](https://linkedin.com/in/raffaelyg)
- **GitHub:** [github.com/raffaelyg](https://github.com/raffaelyg)
