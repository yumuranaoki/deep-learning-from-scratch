import sys
import os
import numpy as np

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))

from common.util import processing, create_co_matrix, cos_similarity, most_similar, create_ppmi_matrix

text = 'You say goodbye and I say Hello.'
corpus, word_to_id, id_to_word = processing(text)
print(corpus)
co_matrix = create_co_matrix(corpus, word_to_id)
print(co_matrix)
print(most_similar('you', word_to_id, id_to_word, co_matrix))
ppmi_matrix = create_ppmi_matrix(co_matrix)

# svd
U, S, V = np.linalg.svd(ppmi_matrix)