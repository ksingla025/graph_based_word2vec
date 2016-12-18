#!/usr/bin/python

import glob,commands
import os
from code.path import *
import gzip
import re
import time
import io
import sys
import argparse
import json
from collections import defaultdict
from itertools import izip
from networkx.readwrite import json_graph
import networkx as nx
reload(sys)
sys.setdefaultencoding('utf-8')

''' data_files should be in format

name(s) : 20151028.bg__20151028.en
sentences in file(s) : 
	sent_lang1 ||| sent_lang2
'''
class CreateDictionary(object):

	def __init__(self):

		commands.getstatusoutput("mkdir -p")
		self.data_files = glob.glob(DATA_PARALLEL+"*")

	'''generate alignments'''
	def get_alignments(self):

		for file in self.data_files:
			''' parse each file using fast_align and generate alignments '''

			file_name = os.path.basename(file)
			
			print "Command :",FAST_ALIGN + " -i " + file + " -v -d -o > "+DATA_ALIGN+file_name+".fastalign"
			commands.getstatusoutput(FAST_ALIGN + " -i " + file + " -v -d -o  > "+DATA_ALIGN+file_name+".fastalign" )
			
			print "Command :",FAST_ALIGN + " -i " + file + " -r -v -d -o  > "+DATA_ALIGN+file_name+".reversefastalign" 
			commands.getstatusoutput(FAST_ALIGN + " -i " + file + " -r -v -d -o  > "+DATA_ALIGN+file_name+".reversefastalign" )
			
			print "Command :",ALIGN_UTIL + " -i " + DATA_ALIGN+file_name+\
				".fastalign -j " + DATA_ALIGN+file_name+".reversefastalign -c intersect > "+\
				DATA_ALIGN+file_name+".symfastalign"
			commands.getstatusoutput(ALIGN_UTIL + " -i " + DATA_ALIGN+file_name+\
			".fastalign -j " + DATA_ALIGN+file_name+".reversefastalign -c intersect > "+\
			DATA_ALIGN+file_name+".symfastalign")

	''' read all alignments and a create weighted dictionaries '''
	def alignments_to_dic(self):

		for file in self.data_files:

			file_name = os.path.basename(file)

			output_forward_dictionary = DATA_DICT+file_name+".fwdict"
			output_backward_dictionary = DATA_DICT+file_name+".bwdict"
			output_product_dictionary = DATA_DICT+file_name+".dict"

			# open input files
			input_parallel_file, input_alignments_file = open(file, mode='r'),\
				open(DATA_ALIGN+file_name+".symfastalign", mode='r')

			# initialize dictionaries
			forward_dict, backward_dict = defaultdict(lambda: defaultdict(float)),\
				defaultdict(lambda: defaultdict(float))

			# read one sentence pair at a time
			sentence_pairs_count = -1
			for sentence_pair_line, alignments_line in izip(input_parallel_file,input_alignments_file):
				# read the source and target tokens of this sentence pair
				sentence_pairs_count += 1
				sentence_pair_line, alignments_line = sentence_pair_line.rstrip(), alignments_line.rstrip()
				source_line, target_line = sentence_pair_line.split(" ||| ")
				source_tokens, target_tokens = source_line.split(' '), target_line.split(' ')

				# read the translationally-equivalent word pairs according to alignments

				if len(alignments_line.strip()) == 0: continue
				word_pair_indexes = [tuple(x.split('-')) for x in alignments_line.split(' ')]
				word_pairs = [ (source_tokens[int(source_index)], target_tokens[int(target_index)],) for (source_index, target_index,) in word_pair_indexes ]
				#print word_pairs

				for (source_token, target_token) in word_pairs:
					forward_dict[source_token][target_token] += 1
					backward_dict[target_token][source_token] += 1

			input_parallel_file.close()
			input_alignments_file.close()

			# write forward dictionary
			with open(output_forward_dictionary, mode='w') as forward_file:
				for source_word in forward_dict.keys():
					normalizer = sum(forward_dict[source_word].values())
					for target_word, alignments_count in forward_dict[source_word].viewitems():
						prob = 1.0 * alignments_count / normalizer
						forward_dict[source_word][target_word] = prob
						if alignments_count == 1: continue
						forward_file.write(u'{} {} {}\n'.format(source_word, target_word, prob))

			# write backward dictionary
			with open(output_backward_dictionary, mode='w') as backward_file:
				for target_word in backward_dict.keys():
					normalizer = sum(backward_dict[target_word].values())
					for source_word, alignments_count in backward_dict[target_word].viewitems():
						prob = 1.0 * alignments_count / normalizer
						backward_dict[target_word][source_word] = prob
						if alignments_count == 1: continue
						backward_file.write(u'{} {} {}\n'.format(target_word, source_word, prob))

			# write product dictionary (i.e., forward probability * backward probability)
			with open(output_product_dictionary, mode='w') as product_file:
				for source_word in forward_dict.keys():
					for target_word, forward_prob in forward_dict[source_word].viewitems():
						product = forward_prob * backward_dict[target_word][source_word]
						product_file.write(u'{} {} {}\n'.format(source_word, target_word, product))








''' this class uses the dictionary stored at "DATA_DICT" to return a 
searchable connected graph'''

class DictionaryGraph(object):


	def __init__(self):

		self.dict_files = glob.glob(DATA_DICT+"*.dict")
		self.dictionary_tuples = []

	def load_dictionary(self):
		''' load each dictionary entry as dictionary tuple '''

		langs = set()

		for file in self.dict_files:
		
			
			file_name = os.path.basename(file)
			lang1 = file_name.split(".")[1].split("_")[0]
			lang2 = file_name.split(".")[-2]

			langs.add(lang1); langs.add(lang2);

			print 'reading the {}->{} dictionary: {}'.format(lang1, lang2, file_name)

			for line in io.open(file, encoding='utf8', mode='r'):
				self.dictionary_tuples.append( (lang1, lang2, line) )

	def create_graph(self,dictionary):
		''' creates a networkx graph from the dictionary tuples and saves\
		a json graph'''

		edges = []
		translations_counter = 0
		g = nx.Graph()

		for (src_lang, tgt_lang, line) in self.dictionary_tuples:
			if translations_counter % 5000 == 0:
				print 'done processing {} dictionary items (in total)'.format(translations_counter)
			translations_counter += 1

			splits = line.strip().split()

			assert len(splits) == 3

#			print splits[0],splits[1]
#			print dictionary
			try:
				splits[0] = splits[0].encode('utf8')+":ID:"+src_lang
				splits[1] = splits[1].encode('utf8')+":ID:"+tgt_lang
				src_token = dictionary[splits[0]]
				tgt_token = dictionary[splits[1]]
				prob = float(splits[2].strip())
#				print splits[0],splits[1],prob
				edges.append((src_token,tgt_token,prob))
				edges.append((tgt_token,src_token,prob))
			except:
				continue
#		print "edges",edges
		edges = list(set(edges))
		for edge in edges:
			g.add_edge(edge[0],edge[1],weight=edge[2])

		d = json_graph.node_link_data(g)
		json.dump(d, open(DATA_DICT+"dict_graph.json",'w'))  

	def load_json_file(self,filename):
		''' loads networkx json graph '''

		with open(filename) as f:
			js_graph = json.load(f)
		return json_graph.node_link_graph(js_graph)




