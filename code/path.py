#!/usr/bin/python

import commands

''' this file sets all paths required in the model '''

DATA = "/home/karan/Desktop/projects/graph_based_word2vec/data/"

''' one needs to create a folder 'parallel' in the data folder to keep the path '''
DATA_PARALLEL = DATA+"parallel/"

DATA_ALIGN = DATA+"alignments/"
DATA_MONO = DATA+"mono/"
DATA_DICT = DATA+"dict/"

commands.getstatusoutput("mkdir -p "+DATA_ALIGN)
commands.getstatusoutput("mkdir -p "+DATA_MONO)
commands.getstatusoutput("mkdir -p "+DATA_DICT)


## set path for aligner 
FAST_ALIGN = "/home/karan/Desktop/projects/graph_based_word2vec/tools/fast_align/build/fast_align" 
ALIGN_UTIL = "/home/karan/Desktop/projects/graph_based_word2vec/tools/fast_align/build/atools"
