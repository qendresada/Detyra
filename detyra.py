import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

train_df = pd.read_csv(r"C:\Users\dijar\Desktop\Detyra Linkplus\AGNEWS_TRAIN.csv")
test_df  = pd.read_csv(r"C:\Users\dijar\Desktop\Detyra Linkplus\AGNEWS_TEST.csv")

train_df["text"] = train_df["Title"] + " " + train_df["Description"]
test_df["text"]  = test_df["Title"] + " " + test_df["Description"]

stopwords = {"the","a","an","is","are","was","were","to","of","in","and","on","at","for","with"}

def clean_text_basic(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords]
    return " ".join(tokens)

train_df["clean_text"] = train_df["text"].apply(clean_text_basic)
test_df["clean_text"]  = test_df["text"].apply(clean_text_basic)

print("Numri i mostrave per kategori (train):")
print(train_df["Class Index"].value_counts())

all_words = " ".join(train_df["clean_text"].sample(2000, random_state=42)).split()
print("Fjalet më të shpeshta:", Counter(all_words).most_common(15))

X_train = train_df["clean_text"]
y_train = train_df["Class Index"]

X_test = test_df["clean_text"]
y_test = test_df["Class Index"]

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Raporti i klasifikimit:\n", classification_report(y_test, y_pred))

train_pred = model.predict(X_train_vec)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("\n==========================")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test  Accuracy: {test_acc*100:.2f}%")
print("==========================")

def predict_category(text):
    clean = clean_text_basic(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    categories = {1:"World", 2:"Sports", 3:"Business", 4:"Sci/Tech"}
    return categories[pred]

print(predict_category("NASA announces new mission to Mars"))
print(predict_category("Barcelona wins the Champions League"))
print(predict_category("Stock markets fall after economic report"))
print(predict_category("New AI technology is transforming education"))


