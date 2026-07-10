import math
from sklearn.feature_extraction.text import TfidfVectorizer
doc=["I love programming in Python.",
    "Python is a great programming language.",
    "I enjoy solving problems with Python."]
tfidf_vectorizer = TfidfVectorizer()
result = tfidf_vectorizer.fit_transform(doc)
print(result.get_feature_names_out)
print(result)