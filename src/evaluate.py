import joblib # You'll need this to save/load your model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    print("--- Model Performance Report ---")
    print(classification_report(y_test, predictions))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

if __name__ == "__main__":
    # You would typically load your trained model here
    # model = joblib.load('models/sentiment_model.pkl')
    # evaluate_model(model, X_test_vec, y_test)
    print("Evaluation module ready. Call this after training.")