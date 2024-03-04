import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(base_model: str, model_path: str, labels_path: str):
    with open(labels_path) as f:
        labels = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )
    model.load_adapter(model_path)
    return tokenizer, model, labels

def predict(text, tokenizer, model, labels, threshold=0.5, return_probs=True):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.sigmoid(logits)

    if return_probs:
        return probs.numpy()

    predictions = (probs > threshold).int()
    predicted_labels = [labels[i] for i in range(len(labels)) if predictions[0][i] == 1]
    return predicted_labels

# Streamlit UI
st.title('AQC Demo')

# Model and labels paths (consider these as fixed for the app)
model_path = 'model_v1'
labels_path = 'class_labels.json'
base_model = 'distilbert-base-uncased' # constant

# Load model and labels
tokenizer, model, labels = load_model(
    base_model=base_model,
    model_path=model_path,
    labels_path=labels_path
)

# User inputs
sentence = st.text_area(
    'Enter a sentence for classification:', 
    'The class material was SO difficult but the TA helped me understand the material much better.'
)
threshold = st.slider('Select a threshold for classification:', 0.0, 1.0, 0.2)

# Predict button
if st.button('Predict'):
    preds = predict(
        text=sentence,
        tokenizer=tokenizer,
        model=model,
        labels=labels,
        threshold=threshold, 
        return_probs=False
    )
    st.write('Predicted labels:')
    st.write(preds)
