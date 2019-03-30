import sys
sys.path.append('..')
import numpy as np

def processing(text):
  text = text.lower()
  text = text.replace('.', ' .')
  words = text.split(' ')

  word_to_id = {}
  id_to_word = {}
  for word in words:
    if word not in word_to_id:
      new_id = len(word_to_id)
      word_to_id[word] = new_id
      id_to_word[new_id] = word

  # idのnp.array
  corpus = np.array([word_to_id[word] for word in words])
  return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, word_to_id, window_size=1):
  corpus_size = len(corpus)
  vocab_size = len(word_to_id) 
  co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
  for idx, word_id in enumerate(corpus):
    for i in range(1, window_size + 1):
      left_idx = idx - i
      right_idx = idx + i

      if left_idx >= 0:
        left_word_id = corpus[left_idx]
        co_matrix[word_id, left_word_id] += 1

      if right_idx < corpus_size:
        right_word_id = corpus[right_idx]
        co_matrix[word_id, right_word_id] += 1

  return co_matrix

def cos_similarity(x, y, eps=1e-8):
  nx = x / (np.sqrt(np.sum(x**2)) + eps)
  ny = y / (np.sqrt(np.sum(y**2)) + eps)
  return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, co_matrix, top=5):
  if query not in word_to_id:
    return
  
  query_id = word_to_id[query]
  query_vec = co_matrix[query_id]

  vocab_size = len(word_to_id)
  similarity = np.zeros(vocab_size)
  for i in range(vocab_size):
    similarity[i] = cos_similarity(query_vec, co_matrix[i])

  count = 0
  rank = {}
  for idx, val in enumerate((-1 * similarity).argsort()):
    word = id_to_word[val]
    if word == query:
      continue
    rank[idx] = word

    count += 1
    if count >= top:
      break
  
  return rank

# 共起行列を引数にとって、正の相互情報量のmatrixを作成
# カウントは共起行列ベースで考える
def create_ppmi_matrix(co_matrix, eps = 1e-8):
  ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float32)
  N = np.sum(co_matrix)
  row_N = np.sum(co_matrix, axis=0)
  for i in range(len(row_N)):
    for j in range(len(row_N)):
      pmi = np.log2(N * co_matrix[i, j] / ((row_N[i] * row_N[j]) + eps))
      ppmi_matrix = max(0, pmi)
  return ppmi_matrix
    
def create_contexts_target(corpus, window_size=1):
  target = corpus[window_size:-window_size]
  context = []
  for i in range(window_size, len(corpus) - window_size):
    context_row = []
    for j in range(-window_size, window_size + 1):
      if j == 0:
        continue
      context_row.append(corpus[i+j])
    context.append(context_row)

  return np.array(context), np.array(target)

def convert_one_hot(corpus, vaocab_size):
  N = corpus.shape[0]
  if corpus.ndim == 1:
    one_hot = np.zeros((N, vaocab_size), dtype=np.int32)
    for i, word_id in enumerate(corpus):
      one_hot[i, word_id] = 1

  if corpus.ndim == 2:
    C = corpus.shape[1]
    one_hot = np.zeros((N, C, vaocab_size), dtype=np.int32)
    for i, contexts in enumerate(corpus):
      for j, word_id in enumerate(contexts):
        one_hot[i, j, word_id] = 1
  
  return one_hot


