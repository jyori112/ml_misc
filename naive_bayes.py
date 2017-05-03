import numpy as np
import csv
from sklearn.metrics import f1_score

def load_data(name='train'):
  with open('data/20news-bydate/{}.data'.format(name), 'r') as f:
    data = f.read().split('\n')

  data = [line.split() for line in data if len(line) > 0]
  data = np.asarray([(int(doc)-1, int(word)-1, int(count)) for doc, word, count in data])

  with open('data/20news-bydate/{}.label'.format(name), 'r') as f:
    labels = f.read().split('\n')

  labels = np.asarray([int(cat)-1 for cat in labels if len(cat) > 0])

  doc = data[:, 0]
  word = data[:, 1]
  count = data[:, 2]
  return labels, doc, word, count

def main():
  V = 61188

  # Load train data
  train_label, train_doc, train_word, train_count = load_data('train')
  print("TRAIN DATA LOADED")

  # Create Count Matrix
  train_count_matrix = np.ones(shape=(np.max(train_label)+1, V))
  for d, w, c in zip(train_doc, train_word, train_count):
    train_count_matrix[train_label[d], w] += c

  # Normalize
  p = train_count_matrix / np.sum(train_count_matrix, axis=1)[:, None]

  # Count each category
  label_count = np.bincount(train_label)

  # Normalize
  pi = label_count/np.sum(label_count)
  print("MODEL LEARNED")

  # Load test data
  test_label, test_doc, test_word, test_count = load_data('test')
  N = len(test_label)
  print("TEST DATA LOADED")

  # Make prediction
  pred_cat = np.ndarray(shape=(N, ), dtype=np.int32)
  for doc in range(N):
    words = test_word[test_doc == doc]
    counts = test_count[test_doc == doc]
    scores = np.log(p[:, words]).dot(counts)+np.log(pi)
    pred_cat[doc] = np.argmax(scores)

  print(f1_score(pred_cat, test_label, average='macro'))

if __name__ == '__main__':
  main()
