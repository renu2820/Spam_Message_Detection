# 📧 Email Spam Detection

A machine learning project to classify emails as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** techniques. The project applies algorithms like **Naïve Bayes, Logistic Regression, and SVM**, and evaluates performance using **Accuracy, Precision, Recall, and F1-Score**.

---

## 🚀 Features
- Preprocesses raw email text with **tokenization, stopword removal, and TF-IDF**  
- Implements multiple ML models for classification  
- Evaluates models with key performance metrics  
- Enhances email security by filtering spam and malicious content  

---

## 🛠 Tech Stack
- **Python**  
- **Scikit-learn**  
- **NLTK / SpaCy**  
- **Pandas & NumPy**  
- **Jupyter Notebook**  

---

## 📂 Project Structure
├── data/ # Dataset files
├── notebooks/ # Jupyter notebooks for training & testing
├── src/ # Python source code
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 📊 Workflow
1. **Data Preprocessing** – Clean and prepare email text (tokenization, stopword removal, lemmatization).  
2. **Feature Extraction** – Convert text to numerical features using **TF-IDF**.  
3. **Model Training** – Train classifiers like Naïve Bayes, Logistic Regression, and SVM.  
4. **Evaluation** – Assess models using **Accuracy, Precision, Recall, and F1-score**.  

---

## 📈 Results
- Naïve Bayes: ~XX% accuracy  
- Logistic Regression: ~XX% accuracy  
- SVM: ~XX% accuracy  

*(Replace XX% with your actual results after training.)*

---

## ▶️ How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/email-spam-detection.git
   cd email-spam-detection

pip install -r requirements.txt
