import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


documents = [
    "I love machine learning.",
    "Natural language processing is fascinating.",
    "Text vectorization is essential for NLP."
]

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

processed_docs = [preprocess(doc) for doc in documents]
print("Обработанные документы:", processed_docs)

# bag of words
# Текст представляется как набор уникальных слов с их частотами.
# Игнорируется порядок слов, учитывается только частота.
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(processed_docs)
print("BoW матрица:\n", bow_matrix.toarray())
print("Словарь:", vectorizer.get_feature_names_out())

# TF-IDF
# Учитывает важность слова в документе и корпусе.
# Уменьшает вес частых слов (например, предлогов).
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)
print("TF-IDF матрица:\n", tfidf_matrix.toarray().round(2))

# Создает плотные векторные представления слов, учитывая их контекст.
# Позволяет находить семантические связи между словами.
# Токенизация для Word2Vec
sentences = [word_tokenize(doc) for doc in processed_docs]

# Обучение модели
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

# Пример вектора слова "machine"
print("Вектор слова 'machine':", model.wv['machine'])
