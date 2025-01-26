# Automated Job Advertisement Classification Using NLP

## Overview  
This project focuses on developing an **automated job classification system** using **Natural Language Processing (NLP)** techniques. The system processes job advertisements, extracts relevant features, and classifies them into predefined categories to improve accuracy in job postings. The project includes text preprocessing, feature extraction, and machine learning-based classification models.

## Problem Statement  
Job advertisements are often misclassified due to human errors in manual category selection. This project aims to:  
- Automate job category predictions based on textual descriptions.  
- Improve classification accuracy using NLP techniques.  
- Enhance job exposure to relevant candidates.  

## Data Collection  
- **Dataset:** Contains ~750 job advertisements categorized into **Accounting & Finance, Engineering, Healthcare & Nursing, and Sales**.  
- **Data Structure:** Each job ad is stored in text files with the title, job description, and relevant metadata.  

## NLP Pipeline  
### **1. Text Preprocessing**  
- Tokenization using regex patterns.  
- Lowercasing and whitespace removal.  
- Removal of stopwords and highly frequent/rare words.  
- Lemmatization and stemming for better word normalization.  
- Vocabulary generation with indexed word mappings.  

### **2. Feature Engineering**  
Three different types of feature representations were generated:  
- **Bag-of-Words (Count Vectorization)**  
- **TF-IDF Weighted Embeddings**  
- **Pretrained Word Embeddings (e.g., FastText, Word2Vec, GloVe)**  

### **3. Job Classification Models**  
- Implemented **Logistic Regression** and **other supervised learning models** to classify job advertisements based on text features.  
- Compared different language models to evaluate classification accuracy.  
- Conducted **5-fold cross-validation** to ensure robust model evaluation.  

### **4. Model Evaluation & Insights**  
- **Accuracy comparison** between models trained on different feature representations.  
- Analyzed the impact of additional metadata (e.g., job title) on classification performance.  
- Developed **classification reports** and **visualizations** to interpret model results.  

## Implementation  
- **Technologies Used:**  
  - **Python**: NLP processing, modeling, and evaluation.  
  - **Jupyter Notebooks**: Interactive analysis and experimentation.  
  - **Pandas, NumPy, Scikit-learn**: Data handling and machine learning.  
  - **NLTK, SpaCy, Gensim**: Natural Language Processing.  
  - **Matplotlib, Seaborn**: Data visualization.  
