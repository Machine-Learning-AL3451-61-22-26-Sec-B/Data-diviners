import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Define a function to train and evaluate the Naive Bayes classifier
def train_and_evaluate_naive_bayes(data):
    # Preprocessing
    data['labelnum'] = data.label.map({'pos': 1, 'neg': 0})
    X = data.message
    y = data.labelnum

    # Split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature extraction
    count_v = CountVectorizer()
    Xtrain_dm = count_v.fit_transform(Xtrain)
    Xtest_dm = count_v.transform(Xtest)

    # Training the model
    clf = MultinomialNB()
    clf.fit(Xtrain_dm, ytrain)
    pred = clf.predict(Xtest_dm)

    # Calculate accuracy metrics
    accuracy = accuracy_score(ytest, pred)
    recall = recall_score(ytest, pred)
    precision = precision_score(ytest, pred)
    conf_matrix = confusion_matrix(ytest, pred)

    return accuracy, recall, precision, conf_matrix, Xtest, pred

# Streamlit app UI
st.title("Text Classification with Naive Bayes")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    msg = pd.read_csv(uploaded_file, names=['message', 'label'])
    st.write("Total Instances of Dataset: ", msg.shape[0])

    # Train and evaluate the Naive Bayes classifier
    accuracy, recall, precision, conf_matrix, Xtest, pred = train_and_evaluate_naive_bayes(msg)

    # Display results
    st.write("Accuracy Metrics:")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Recall: {recall}")
    st.write(f"Precision: {precision}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    # Display predictions
    st.write("Predictions on Test Data:")
    predictions = pd.DataFrame({'Message': Xtest, 'Predicted Label': ['pos' if p == 1 else 'neg' for p in pred]})
    st.write(predictions)

# To run the streamlit app, save this file as app.py and run `streamlit run app.py` in the terminal.
