from vstore import VectorStore
import numpy as np

vector_store = VectorStore()
corpus = []
vocabulary = set()
word_to_index = {}

def update_vocabulary(sentences):
    global vocabulary, word_to_index
    for sentence in sentences:
        tokens = sentence.lower().split()
        vocabulary.update(tokens)
    word_to_index = {word: i for i, word in enumerate(vocabulary)}


def vectorize_sentence(sentence):
    vector = np.zeros(len(vocabulary))
    tokens = sentence.lower().split()
    for token in tokens:
        if token in word_to_index:
            vector[word_to_index[token]] += 1
    return vector

def add_sentences_to_vector_store(sentences):
    global corpus
    corpus.extend(sentences)
    for sentence in sentences:
        vector = vectorize_sentence(sentence)
        vector_store.add_vector(sentence, vector)

def find_similar_sentences(query_sentence, num_results=2):
    query_vector = vectorize_sentence(query_sentence)
    similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=num_results)
    print("Query Sentence:", query_sentence)
    print("Similar Sentences:")
    for sentence, similarity in similar_sentences:
        print(f"{sentence}: Similarity = {similarity:.4f}")


initial_sentences = [
    "Friendship is everything. Friendship is more than talent. It is more than the government. It is almost the equal of family",
    "Great men are not born great, they grow great",
    "Never hate your enemies. It affects your judgment",
    "Revenge is a dish that tastes best when served cold"
]

update_vocabulary(initial_sentences)

add_sentences_to_vector_store(initial_sentences)

query_sentence = "is a dish "
find_similar_sentences(query_sentence)
