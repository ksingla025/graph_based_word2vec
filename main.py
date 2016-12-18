#!/usr/bin/python

''' Graph-based-word2vec for multiple languages '''

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import collections
import math
import os
import random
import zipfile
import glob
from itertools import compress

from networkx.readwrite import json_graph
import networkx as nx
import numpy as np
import tensorflow as tf

from code.path import *
from code.dicgraph import CreateDictionary,DictionaryGraph

# Read the data into a list of strings.
def read_data():
	"""Extract the first file enclosed in a zip file as a list of words"""

	data_files = glob.glob(DATA_MONO+"*")

	data = []
	for filename in data_files:
		print filename
		f = open(filename,'r')
		lang_ext = ":ID:" + filename.split(".")[-1]
		for line in f:
			tokens = [x + lang_ext for x in line.split()]
			for i in range(0,len(tokens)):
				tokens[i] = tokens[i]
			data.extend(tokens)
#		with zipfile.ZipFile(filename) as f:
#			data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

# Step 2: Build the dictionary and replace rare words with UNK token.

def build_dataset(words, vocabulary_size=200000):
	'''
	Build the dictionary and replace rare words with UNK token.
	
	Parameters
	----------
	words: list of tokens
	vocabulary_size: maximum number of top occurring tokens to produce, 
		rare tokens will be replaced by 'UNK'
	'''
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict() # {word: index}
	for word, _ in count:
		word = word
		dictionary[word] = len(dictionary)
		data = list() # collect index
		unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count # list of tuples (word, count)
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

# create alignments, create dictionary, and then create graph
def generate_graph(dictionary):
	
	createdic = CreateDictionary()
#	createdic.get_alignments()
	createdic.alignments_to_dic()

	graph = DictionaryGraph()
	graph.load_dictionary()
	graph.create_graph(dictionary)

def load_graph(filename):

	graph = DictionaryGraph()
	load_graph = graph.load_json_file(filename)

	return load_graph
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_multilingual_batch(graph, batch_size, num_skips, skip_window):
  global data_index

  index = data_index - 1
  batch = []
  labels = []
  weights = []
  for i in range(0,batch_size):
  	index = index + 1
  	if data[index+1] in graph.nodes():
  		print graph[data[index+1]]
  		graph_context = graph[data[index+1]]
  		batch.append(graph_context.keys())
  		labels.append(data[index+1])

  		weight_list = []
  		for value in graph_context.values():
  			weight_list.append(value['weight'])
  		weights.append(weight_list)
#  else:
#  	print "no multi-lingual context"


  print batch,weights
  return np.array(batch), np.array(labels), np.array(weights)

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  weights = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
      weights[i * num_skips + j, 0] = 1
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels, weights

def generate_batch_cbow(data, batch_size, num_skips, skip_window):
	'''
	Batch generator for CBOW (Continuous Bag of Words).
	batch should be a shape of (batch_size, num_skips)
	Parameters
	----------
	data: list of index of words
	batch_size: number of words in each mini-batch
	num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
	skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
	'''
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1 # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span) # used for collecting data[data_index] in the sliding window
	# collect the first window of words
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	# move the sliding window  
	for i in range(batch_size):
		mask = [1] * span
		mask[skip_window] = 0 
		batch[i, :] = list(compress(buffer, mask)) # all surrounding words
		labels[i, 0] = buffer[skip_window] # the word at the center 
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels
#def generate_batch_cbow(data,batch_size,num_skips, skip_window):

#def get_graph_neighbours(data,ndepth=1,weighted='yes'):

#class Word2Vec(BaseEstimator, TransformerMixin):

if __name__ == '__main__':

	print "loading Monolingual Data"

	words = read_data()
	
	print('Data size', len(words))

	data, count, dictionary, reverse_dictionary = build_dataset(words)

	#generates dictionary from parallel data and saves graph
	generate_graph(dictionary)

#	for i in dictionary.keys()[:20]:
#		print i

	#loads graph
	graph = load_graph("/home/karan/Desktop/projects/graph_based_word2vec/data/dict/dict_graph.json")
	
	print('Words in Graph',len(graph.nodes()))

	del words  # Hint to reduce memory.

	print('Most common words (+UNK)', count[:50])
	print('Sample data', data[:20], [reverse_dictionary[i] for i in data[:10]])

	
	data_index = 0

	while data_index < 200:
		batch, labels, weights = generate_multilingual_batch(graph,batch_size=2, num_skips=2, skip_window=1)

		for i in range(batch.shape[0]):
			print(batch[i],weights[i],
				'->', labels[i], reverse_dictionary[labels[i]])

		batch, labels = generate_batch_cbow(data,batch_size=2, num_skips=2, skip_window=1)
		for i in range(2):
			print(batch[i,:], reverse_dictionary[batch[i,0]],
				'->', labels[i, 0], reverse_dictionary[labels[i, 0]])







