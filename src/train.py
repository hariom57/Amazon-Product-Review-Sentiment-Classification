from preprocess import load_and_clean_data, get_splits, get_vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Execute Pipeline
df = load_and_clean_data()
X_train, X_test, y_train, y_test = get_splits(df)

vectorizer = get_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Simple export of results
print(f"Model Training Complete.")
print(f"F1 Score: {f1_score(y_test, model.predict(X_test_vec), average='weighted'):.4f}")

####################################################################################
# EVALUTION
####################################################################################

import joblib
import os

# Create a 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model and the vectorizer
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model saved to models/ folder.")