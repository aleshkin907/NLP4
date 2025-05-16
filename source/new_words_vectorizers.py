from transformers import AutoTokenizer, AutoModel
import torch

# Загрузка предобученной модели и токенизатора
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Пример текста
texts = ["Hello, world!", "BERT is powerful."]

# Токенизация и преобразование в тензоры
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Получение эмбеддингов
with torch.no_grad():
    outputs = model(**inputs)

# Извлечение эмбеддингов предложений (усреднение токенов)
embeddings = torch.mean(outputs.last_hidden_state, dim=1)
print("BERT embeddings shape:", embeddings.shape)

import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Конвертация GloVe в формат Word2Vec
glove_input = "../assets/glove.6B.100d.txt"
word2vec_output = "glove.6B.100d.word2vec.txt"
glove2word2vec(glove_input, word2vec_output)

# Загрузка модели
model = KeyedVectors.load_word2vec_format(word2vec_output, binary=False)

# Пример получения вектора слова
vector = model["computer"]
print("GloVe вектор слова 'computer':", vector)

# Эмбеддинг предложения (усреднение векторов слов)
sentence = "machine learning is awesome"
words = sentence.split()
vectors = [model[word] for word in words if word in model]
sentence_vector = np.mean(vectors, axis=0)
print("Средний вектор предложения:", sentence_vector)

from gensim.models import FastText

# Загрузка предобученной модели
model = FastText.load_fasttext_format("wiki-news-300d-1M.vec")

# Вектор слова (даже для OOV-слов!)
vector = model.wv["artificial"]
print("FastText вектор слова 'artificial':", vector)

# Эмбеддинг предложения
sentence = "deep learning models"
vectors = [model.wv[word] for word in sentence.split()]
sentence_vector = np.mean(vectors, axis=0)

from gensim.models import FastText

# Пример данных
sentences = [
    ["machine", "learning", "is", "cool"],
    ["natural", "language", "processing"]
]

# Обучение модели
model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Вектор для OOV-слова (например, "machinne")
vector = model.wv["machinne"]
