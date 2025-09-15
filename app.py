import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and clean the dataset
data = pd.read_csv(r"spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not spam', 'spam'])

# Prepare data
X = data['Message']
y = data['Category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text data to numerical features
vectorizer = CountVectorizer(stop_words='english')
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_features, y_train)

# Define prediction function
def predict_spam(message):
    message_features = vectorizer.transform([message])
    prediction = model.predict(message_features)[0]
    return f"Prediction: {prediction}"

# Launch Gradio interface
gr.Interface(
    fn=predict_spam,
    inputs="text",
    outputs="text",
    title="📩 Spam Detection with Gradio",
    description="Enter a message and the model will predict whether it's spam or not."
).launch()
