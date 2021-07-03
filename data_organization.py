import collections
import nltk
import os
import random
import collections  # 1.5
import multiprocessing as mp  # 1.2
import re  # 1.3

def load_files(directory):
    result = []
    fileExt = r".cats"
    no_of_class = [2]
    for fname in os.listdir(directory) :
        if fname.endswith(fileExt):
            with open(directory + '/' + fname, 'r', encoding='ISO-8859-1') as f:
                print(f.read)
                for i in no_of_class:
                    word_matcher = re.compile(r""+str(i)+",\d,")
                    for line in f :
                        for match in re.finditer(word_matcher,line):
                            print(line)

    return result


positive_examples = load_files('enron_classifier_organized_data/2')
