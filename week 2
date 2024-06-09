import streamlit as st
import numpy as np
from collections import Counter

class Node:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, value=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X, y, attribute, threshold):
    left_indices = X[:, attribute] < threshold
    left_y = y[left_indices]
    right_y = y[~left_indices]
    p_left = len(left_y) / len(y)
    p_right = len(right_y) / len(y)
    return entropy(y) - p_left * entropy(left_y) - p_right * entropy(right_y)

def find_best_split(X, y):
    best_gain = 0
    best_attribute = None
    best_threshold = None
    for attribute in range(X.shape[1]):
        thresholds = np.unique(X[:, attribute])
        for threshold in thresholds:
            gain = information_gain(X, y, attribute, threshold)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
                best_threshold = threshold
    return best_attribute, best_threshold

def build_tree(X, y, max_depth=None):
    if max_depth == 0 or len(np.unique(y)) == 1:
        return Node(value=Counter(y).most_common(1)[0][0])

    best_attribute, best_threshold = find_best_split(X, y)
    if best_threshold is None:
        return Node(value=Counter(y).most_common(1)[0][0])

    left_indices = X[:, best_attribute] < best_threshold
    left_X, left_y = X[left_indices], y[left_indices]
    right_X, right_y = X[~left_indices], y[~left_indices]
    left_node = build_tree(left_X, left_y, max_depth - 1 if max_depth else None)
    right_node = build_tree(right_X, right_y, max_depth - 1 if max_depth else None)
    return Node(attribute=best_attribute, threshold=best_threshold, left=left_node, right=right_node)

def predict_sample(x, tree):
    if tree.value is not None:
        return tree.value
    if x[tree.attribute] < tree.threshold:
        return predict_sample(x, tree.left)
    else:
        return predict_sample(x, tree.right)

def predict(X, tree):
    return [predict_sample(x, tree) for x in X]

def main():
    st.title("ID3 Decision Tree Classifier")
    st.sidebar.title("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = np.genfromtxt(uploaded_file, delimiter=',', skip_header=1)
        X = data[:, :-1]
        y = data[:, -1]

        max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=10, value=5)
        tree = build_tree(X, y, max_depth=max_depth)
        st.write("Decision Tree Built.")
        
        st.subheader("Make Predictions")
        input_data = st.text_input("Enter input data (comma-separated values):")
        if input_data:
            input_values = np.array([float(x.strip()) for x in input_data.split(',')]).reshape(1, -1)
            prediction = predict(input_values, tree)
            st.write(f"Prediction: {prediction[0]}")

if __name__ == "__main__":
    main()
