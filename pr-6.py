from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

texts = [
    'купить скидка акция бесплатно',
    'распродажа лучшие цены',
    'привет как дела',
    'встретимся завтра по проекту',
    'выиграй миллион рублей',
    'лекция перенесена',
]
labels = [1, 1, 0, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print('Accuracy:', accuracy_score(y_test, preds))