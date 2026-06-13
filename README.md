# 🏆 Smart Review Analyzer

### Predicting Amazon Product Ratings from Review Text using NLP & Multi-class Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![TF-IDF](https://img.shields.io/badge/TF--IDF-Feature%20Engineering-yellow)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-brightgreen?logo=google-colab)

> **Amazon ML Summer School 2026** — Application Project  
> **Author**: Abhishek Singh Rajawat  
> **GitHub**: [Benjamintennyson27](https://github.com/Benjamintennyson27) | **Kaggle**: [benjamin10nyson](https://www.kaggle.com/benjamin10nyson)

---

## 📌 Problem Statement

Amazon processes **billions of product reviews** daily. Each review has a **star rating (1–5)** and **text content**. This project builds an end-to-end **multi-class NLP classification pipeline** that:

1. Takes raw review text as input
2. **Predicts the exact star rating (1 to 5)** — not just positive/negative
3. Analyzes which words and patterns are most predictive of each rating level

### Why Multi-Class (5 stars) Instead of Binary Sentiment?
- Binary sentiment (positive/negative) is a **beginner-level** task
- Real-world Amazon needs to distinguish between a 1-star angry review vs a 3-star "it's okay" vs a 5-star rave
- Multi-class is significantly harder — shows **real ML maturity**

---

## 📊 Dataset

| Field | Details |
|---|---|
| **Name** | Amazon Fine Food Reviews |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| **Size** | ~568,454 reviews (~300 MB) |
| **Features** | Review Text, Star Rating (1–5), Summary, Helpfulness Score |
| **Target** | `Score` — Star Rating (1, 2, 3, 4, or 5) |

### Rating Distribution
| Rating | Approximate % | Challenge |
|---|---|---|
| ⭐ 1 | ~10% | Minority class |
| ⭐⭐ 2 | ~6% | **Rarest — hardest to predict** |
| ⭐⭐⭐ 3 | ~8% | Ambiguous language |
| ⭐⭐⭐⭐ 4 | ~18% | Often confused with 5-star |
| ⭐⭐⭐⭐⭐ 5 | ~58% | Majority class — risk of bias |

> **Key Challenge**: Heavy class imbalance — the model must not just predict "5 stars" for everything.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **ML Framework** | Scikit-learn |
| **NLP** | NLTK, TF-IDF Vectorizer |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Class Imbalance** | Scikit-learn `class_weight='balanced'` |
| **Platform** | Google Colab |

---

## 🔧 Methodology

### Pipeline Overview
```
Raw Text → Cleaning → Tokenization → Stemming → TF-IDF → ML Models → Predictions
```

### Phase-by-Phase

| Phase | Description |
|---|---|
| **1. Data Loading** | Load 568K reviews, remove duplicates, handle missing values |
| **2. EDA** | Rating distribution, review length analysis, class imbalance visualization |
| **3. Text Preprocessing** | Lowercase, remove HTML/URLs, tokenize, remove stopwords, stem (Porter) |
| **4. Feature Engineering** | TF-IDF with 10K features, unigrams + bigrams, sublinear TF |
| **5. Model Training** | 4 classifiers with balanced class weights |
| **6. Evaluation** | Accuracy, Weighted F1, Confusion Matrix, Per-class metrics |
| **7. Feature Analysis** | Top predictive words per rating, word clouds |

### Models Compared

| Model | Why Include It |
|---|---|
| **Logistic Regression** | Strong baseline for text classification, fast, interpretable |
| **Multinomial Naive Bayes** | Classic NLP model, works well with TF-IDF |
| **Random Forest** | Non-linear, captures complex patterns |
| **Linear SVM** | Excellent for high-dimensional text data |

---

## 📈 Results

| Metric | Value | Notes |
|---|---|---|
| **Overall Accuracy** | ~65-70% | 5-class prediction from subjective text |
| **Weighted F1** | ~0.64-0.68 | Primary evaluation metric |
| **1-Star Precision** | ~75%+ | Strong negative words are distinctive |
| **5-Star Recall** | ~80%+ | Majority class, well-represented |
| **2/3-Star F1** | ~35-50% | Hardest — ambiguous language |

> **Why ~65% accuracy is impressive**: Distinguishing 5 classes of subjective human opinion is extremely challenging. Even humans can't always tell if a review is 3-star vs 4-star.

---

## 📊 Visualizations

| Visualization | Description |
|---|---|
| Rating Distribution | Shows class imbalance across 5 star levels |
| Review Length Analysis | Box plots and word counts by rating |
| Model Comparison | Side-by-side accuracy and F1 comparison |
| Confusion Matrix | Prediction patterns and common errors |
| Per-Class Metrics | Precision, Recall, F1 per star rating |
| Feature Importance | Top 15 most predictive words per class |
| Word Clouds | Visual word frequency by rating level |

---

## 📁 Project Structure

```
smart-review-analyzer/
├── README.md                          # This file
├── Smart_Review_Analyzer.ipynb        # Main Colab notebook (full pipeline)
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── images/                            # Generated visualizations
│   ├── rating_distribution.png
│   ├── review_length_boxplot.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── per_class_metrics.png
│   ├── top_features_per_class.png
│   ├── wordcloud_1star.png
│   ├── wordcloud_2star.png
│   ├── wordcloud_3star.png
│   ├── wordcloud_4star.png
│   ├── wordcloud_5star.png
│   └── wordclouds_all.png
│
└── results/
    └── classification_report.txt      # Full classification report
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open `Smart_Review_Analyzer.ipynb` in [Google Colab](https://colab.research.google.com/)
2. The notebook will automatically download the dataset via Kaggle API
3. Run all cells — takes ~10-15 minutes total

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/Benjamintennyson27/smart-review-analyzer.git
cd smart-review-analyzer

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place Reviews.csv in the data/ folder

# Run the notebook
jupyter notebook Smart_Review_Analyzer.ipynb
```

---

## 🧠 Skills Demonstrated

| Skill | How It's Shown |
|---|---|
| **NLP Proficiency** | Full text preprocessing pipeline (tokenization, stemming, TF-IDF, n-grams) |
| **Multi-class Classification** | 5-class prediction, not basic binary |
| **Class Imbalance Handling** | Balanced class weights, stratified splitting |
| **Model Comparison** | 4 different algorithms compared systematically |
| **Evaluation Rigor** | Weighted F1, confusion matrix, per-class metrics |
| **Feature Analysis** | Interpretable word-importance analysis per rating |
| **Data Science Pipeline** | End-to-end: EDA → Preprocessing → Training → Evaluation → Visualization |
| **Domain Relevance** | Directly applicable to Amazon's review ecosystem |

---

## 🔗 References

- Dataset: [Amazon Fine Food Reviews — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Scikit-learn: [scikit-learn.org](https://scikit-learn.org/)
- NLTK: [nltk.org](https://www.nltk.org/)
- TF-IDF Theory: [Stanford NLP](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-1.html)

---

## 👤 Author

**Abhishek Singh Rajawat**  
- 📧 abhishekrajawat1808@gmail.com  
- 🔗 [LinkedIn](https://www.linkedin.com/in/abhishek-singh-rajawat-1237573a8/)  
- 🐙 [GitHub](https://github.com/Benjamintennyson27)  
- 📊 [Kaggle](https://www.kaggle.com/benjamin10nyson)  
- 🎓 B.Tech — Jabalpur Engineering College (JEC), 2024–2028

---

*Built for Amazon ML Summer School 2026 Application*
