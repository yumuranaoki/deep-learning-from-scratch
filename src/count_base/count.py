import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))

from common.util import processing, create_co_matrix, cos_similarity, most_similar

text = 'You say goodbye and I say Hello.'
corpus, word_to_id, id_to_word = processing(text)
print(corpus)
co_matrix = create_co_matrix(corpus, word_to_id)
print(co_matrix)
print(most_similar('you', word_to_id, id_to_word, co_matrix))
