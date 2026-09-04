import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load the dataset
# ==========================================

df = pd.read_csv("data/Resume.csv")

print("==========================================")
print("       RESUME SCREENING SYSTEM")
print("==========================================")

print("\nDataset loaded successfully!")
print("Total resumes:", len(df))


# ==========================================
# 2. Clean resume text
# ==========================================

def clean_text(text):
    text = str(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Apply cleaning
df["clean_resume"] = df["Resume_str"].apply(clean_text)

print("\nResume text cleaning completed!")


# ==========================================
# 3. Convert resume text into TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_resume"])

# Target category
y = df["Category"]

print("\nTF-IDF conversion completed!")
print("Number of resumes:", X.shape[0])
print("Number of features:", X.shape[1])


# ==========================================
# 4. Split data into training and testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining resumes:", X_train.shape[0])
print("Testing resumes:", X_test.shape[0])


# ==========================================
# 5. Train Logistic Regression model
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel training completed!")


# ==========================================
# 6. Evaluate the model
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("             MODEL RESULTS")
print("==========================================")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 7. Classification report
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 8. Test with a new resume
# ==========================================

new_resume = """
Python developer with experience in machine learning,
data analysis, pandas, NumPy, scikit-learn, SQL,
Python programming and artificial intelligence.
"""


# Clean new resume
cleaned_resume = clean_text(new_resume)


# Convert new resume using trained TF-IDF
new_resume_vector = vectorizer.transform([cleaned_resume])


# Predict category
prediction = model.predict(new_resume_vector)


# ==========================================
# 9. Display prediction
# ==========================================

print("\n==========================================")
print("       NEW RESUME SCREENING RESULT")
print("==========================================")

print("\nPredicted Category:", prediction[0])

print("\n==========================================")
print("             SCREENING COMPLETE")
print("==========================================")