# 🎬 CineMatch — Movie Recommendation System

A **Content-Based Movie Recommendation System** built using Python, Machine Learning, and Streamlit. This project recommends movies similar to a selected title by analyzing metadata features like genres, keywords, cast, director, and plot overview.

---

## 📸 Overview

| Feature | Description |
|---------|-------------|
| **Algorithm** | Content-Based Filtering using Cosine Similarity |
| **Vectorization** | CountVectorizer (Bag of Words, 5000 features) |
| **Stemming** | NLTK PorterStemmer for vocabulary normalization |
| **Dataset** | TMDB 5000 Movies + Credits (4806 movies) |
| **Frontend** | Streamlit with custom dark-mode glassmorphism UI |
| **Posters** | Fetched dynamically from TMDB API (v3) |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** — Data manipulation and cleaning
- **NumPy** — Numerical operations
- **Scikit-learn** — CountVectorizer and Cosine Similarity
- **NLTK** — Text stemming with PorterStemmer
- **Streamlit** — Interactive web application framework
- **Requests** — HTTP requests for TMDB poster API
- **Pickle** — Serialization of model artifacts

---

## 📂 Project Structure

```
MovieRecommendationSystem/
│
├── app.py                  # Streamlit web application
├── recommendation.py       # Recommendation engine (model loading & inference)
├── preprocess.py           # Data pipeline (download, clean, vectorize, save)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation (this file)
├── movies.pkl              # Preprocessed movies DataFrame (generated)
├── similarity.pkl          # Cosine similarity matrix (generated)
│
├── dataset/
│   └── movies.csv          # Cleaned processed dataset (generated)
│
├── assets/
│   └── no_poster.png       # Fallback poster placeholder image
│
└── notebooks/
    └── recommendation.ipynb # Step-by-step Jupyter notebook walkthrough
```

---

## 🚀 Setup Instructions

### Step 1: Clone or Navigate to the Project

```bash
cd MovieRecommendationSystem
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Preprocessing Pipeline

This script downloads the TMDB 5000 dataset from Hugging Face, cleans data, extracts features, applies stemming, performs vectorization, computes cosine similarity, and saves `.pkl` files.

```bash
python preprocess.py
```

**Expected output:**
```
=== Starting Movie Recommendation System Data Pipeline ===
[Info] Downloading from ...
[1/8] Loading raw movies and credits datasets...
...
[8/8] Calculating Cosine Similarity Matrix...
[Success] Saved movies pickle to: movies.pkl
[Success] Saved similarity matrix pickle to: similarity.pkl
=== Pipeline Execution Completed Successfully ===
```

### Step 4: Launch the Streamlit App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 How the Recommendation Engine Works

### Machine Learning Pipeline

1. **Load Dataset** — Two CSV files: movies metadata + credits (cast/crew).
2. **Merge** — Combine on `title` to get a unified DataFrame.
3. **Clean** — Drop rows with missing values.
4. **Feature Extraction** — Parse JSON columns for genres, keywords, top 3 cast, and director.
5. **Space Collapsing** — Remove spaces in entity names (e.g., `"Christopher Nolan"` → `"ChristopherNolan"`) so multi-word names are treated as single tokens.
6. **Create Tags** — Merge overview + genres + keywords + cast + director into a single text field.
7. **Stemming** — Apply NLTK PorterStemmer to normalize vocabulary (`"loving"` → `"love"`).
8. **Vectorization** — Convert tags to 5000-dimensional vectors using CountVectorizer (Bag of Words).
9. **Cosine Similarity** — Compute pairwise similarity between all 4806 movies.
10. **Recommendation** — For a given movie, sort all others by similarity score and return the top 10.

### Key Concepts Explained

| Concept | Explanation |
|---------|-------------|
| **Content-Based Filtering** | Recommends items similar to what the user already likes, based on item features (not other users' behavior). |
| **Bag of Words (BoW)** | Represents text as a vector of word frequencies, ignoring grammar and word order. |
| **CountVectorizer** | Converts text documents into a matrix of token counts. We use `max_features=5000` to limit the vocabulary. |
| **Cosine Similarity** | Measures the cosine of the angle between two vectors. Score ranges from 0 (completely different) to 1 (identical). |
| **Stemming** | Reduces words to their root form to group similar words (e.g., `running`, `runs`, `ran` → `run`). |

---

## 📖 Viva Preparation — Frequently Asked Questions

### Q1: What type of recommendation system is this?
**A:** This is a **Content-Based Filtering** system. It recommends movies similar to a selected movie by analyzing metadata features like genres, keywords, cast, and director.

### Q2: Why Content-Based and not Collaborative Filtering?
**A:** Content-Based filtering doesn't require user ratings or interaction history — it works purely on movie metadata. Collaborative Filtering needs a user-item interaction matrix, which we don't have in this dataset.

### Q3: Why did you use CountVectorizer instead of TF-IDF?
**A:** Both work well for this use case. CountVectorizer (Bag of Words) counts word frequencies, while TF-IDF weighs words by their importance across documents. For movie tags where most words are already meaningful keywords, CountVectorizer performs similarly to TF-IDF with simpler implementation.

### Q4: Why do you remove spaces from entity names?
**A:** Without this step, `"Sam Worthington"` would be split into `"Sam"` and `"Worthington"`, causing false matches with other actors named Sam. By collapsing to `"SamWorthington"`, each actor becomes a unique token.

### Q5: What is the purpose of stemming?
**A:** Stemming reduces words to their root form. For example, `"loving"`, `"loved"`, `"loves"` all become `"love"`. This ensures that semantically similar words are treated as the same feature, improving recommendation accuracy.

### Q6: How does cosine similarity work?
**A:** Cosine similarity measures the angle between two movie vectors in 5000-dimensional space. If two movies share similar tags, their vectors point in similar directions, yielding a high cosine similarity score (close to 1.0).

### Q7: Why pickle? Can't you just re-run the model?
**A:** Computing the similarity matrix takes time. Saving it with pickle allows the Streamlit app to load the pre-computed matrix instantly, making the web app responsive and fast.

### Q8: What are the limitations?
**A:**
- Only recommends based on metadata — doesn't consider user preferences or ratings.
- Limited to 4806 movies in the TMDB 5000 dataset.
- Poster display depends on TMDB API availability.
- New movies not in the dataset can't be recommended.

---

## 📊 Dataset Information

- **Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (mirrored on Hugging Face)
- **Movies:** 4,806 (after cleaning)
- **Features used:** `id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`
- **Similarity matrix dimensions:** 4806 × 4806

---

## 🎯 Sample Results

**Input:** Avatar

| Rank | Recommended Movie |
|------|------------------|
| 1 | Aliens vs Predator: Requiem |
| 2 | Aliens |
| 3 | Falcon Rising |
| 4 | Independence Day |
| 5 | Titan A.E. |
| 6 | Small Soldiers |
| 7 | Soldier |
| 8 | Battle: Los Angeles |
| 9 | Ender's Game |
| 10 | Aliens in the Attic |

---

## 📝 License

This project is created for educational purposes as part of the **Codec Technology AI/ML Internship**.

---

## 👤 Author

**Abhishek** — AI/ML Intern at Codec Technology
