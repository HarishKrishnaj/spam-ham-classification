# 📩 SMS Spam Classification (TF-IDF + Logistic Regression)

This repository contains a Jupyter Notebook that demonstrates how to build an **SMS Spam Detection system** using **Natural Language Processing (NLP)** techniques and a **Logistic Regression** machine learning model.

The goal of this project is to automatically classify SMS messages as **Spam** or **Ham (Not Spam)** using text preprocessing, feature extraction, and supervised learning.

---

## 🚀 Project Overview

Spam messages are a common problem in communication systems. This project applies classical NLP and machine learning techniques to detect spam messages efficiently.

**Key highlights:**

* Text preprocessing and cleaning
* Feature extraction using **TF-IDF Vectorization**
* Classification using **Logistic Regression**
* Model evaluation and performance analysis

---

## 🧠 Machine Learning Approach

### 1. Data Preprocessing

The raw SMS text is cleaned and prepared using the following steps:

* Removing special characters and punctuation
* Converting text to lowercase
* Tokenization
* Removing stopwords
* Lemmatization

These steps help reduce noise and improve model performance.

### 2. Feature Extraction

* **TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert text data into numerical feature vectors.
* This helps the model understand the importance of words across messages.

### 3. Model Used

* **Logistic Regression**
* Chosen for its simplicity, interpretability, and effectiveness for text classification tasks.

### 4. Model Evaluation

The model is evaluated using standard classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📁 Project Structure

```
SPAM_CLASSIFICATION__DEC_2025.ipynb   # Main Jupyter Notebook
README.md                            # Project documentation
```

---

## 🛠️ Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib / Seaborn (for visualization)
* Jupyter Notebook

---

## ▶️ How to Run the Project

1. Clone the repository or download the notebook
2. Install the required dependencies

   ```bash
   pip install numpy pandas scikit-learn nltk matplotlib seaborn
   ```
3. Open the notebook

   ```bash
   jupyter notebook SPAM_CLASSIFICATION__DEC_2025.ipynb
   ```
4. Run all cells sequentially

---

## 📌 Use Cases

* SMS spam filtering systems
* Email spam detection (with minor modifications)
* NLP learning and experimentation
* Beginner-friendly machine learning project

---

## 📈 Future Improvements

* Try advanced models like Naive Bayes, SVM, or Random Forest
* Use deep learning models (LSTM, BERT)
* Hyperparameter tuning
* Deploy as a web application using Flask or FastAPI

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome. Feel free to fork the project and submit a pull request.

---

## 📜 License

This project is for educational and learning purposes.

---

✨ *Built as a hands-on NLP and Machine Learning project for spam detection.*
