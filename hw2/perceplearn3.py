

import sys
import os
import string
import math
import re
import numpy as np
import random

# Load data from source folder
def load_data(path):

	files_list = [] # Where to find files for parsing

	file_tree =  [fold for fold in os.walk(path)][1:] # 0th item is the current dir 

	# Extract files from all the fold directories
	for fold in file_tree:  
		file_path = fold[0]

		for file in fold[2]:
			files_list.append(file_path+"/"+file)

	text_data = []

	for file in files_list:

		with open(file) as fp:
			text_data.append("".join(fp.readlines()))

	return text_data

# Clean and create dict of tokens for each document
def token_sub(doc_array):
	to_return = []
	global_dict = {}

	map_dict = dict.fromkeys(string.punctuation,"")
	for p in map_dict:
		if p == "|": # This character is used as a delim in output txt files (Highly unlikely to matter)
			map_dict[p] = " "
		else:
			map_dict[p] = " " + p + " "
	mapping = str.maketrans(map_dict)

	for doc in doc_array:
		new_doc = re.sub(r"[0-1]?[0-9](:\d{0,2}((\s*)?am|(\s*)?pm)?|(\s*)?am|(\s*)?pm)", " timetoken ", doc, flags=re.IGNORECASE) # Identify time
		new_doc = re.sub(r"\$\d*(\.\d?\d)?", " moneytoken ", new_doc) # Identify money
		new_doc = re.sub(r"\d+(\.*)?\d*", " digittoken ", new_doc) # Identify digits
		new_doc = re.sub(r"[A-Z]{1}[A-Z,a-z]*,", " introtoken ", new_doc) # Identify "intro words" - Capital word followed by comma (Ex. Formally, )
		new_doc = new_doc.translate(mapping) # Count punctuation as tokens
		new_doc = new_doc.lower() # Make everything lowercase

		new_doc = new_doc.split() # Tokenization

		local_dict = {}

		for token in new_doc:
			if token not in local_dict:
				local_dict[token] = 0
				if token not in global_dict:
					global_dict[token] = 0
				global_dict[token] += 1 # Counts number of docs having token

			local_dict[token] += 1

		to_return.append(local_dict)

	return to_return, global_dict

# Convert dict data into numpy data
def make_fun(features, data, val):
	to_return = []

	for doc in data:
		vec_rep = []
		for word in features:
			if word in doc:
				vec_rep.append(doc[word])
			else:
				vec_rep.append(0)
		
		to_return.append((np.array(vec_rep),val))

	return to_return

# Average perceptron
def train_avg(data, max_iter):

	w = np.random.rand(data[0][0].shape[0])
	b = np.random.rand(1)[0]
	mu = np.zeros(data[0][0].shape[0])
	beta = 0
	c = 0

	acc_counter = 0

	proxy = [i for i in range(0,len(data))]
	random.shuffle(proxy)
	
	for i in range(0, max_iter):
		acc_counter = 0
		for n in proxy:
			x = data[n][0]
			y = data[n][1]

			if y*(np.sum(w*x) + b) <= 0:
				w = w + x*y
				b = b + y
				mu = mu + y*c*x 
				beta = beta + y*c
			else:
				acc_counter += 1

			c += 1

		random.shuffle(proxy)

	#print("Training accuracy: " + str(acc_counter/len(data)))

	return w - (1/c)*mu, b - (1/c)*beta

# Vanilla perceptron
def train_vanilla(data, max_iter):

	w = np.random.rand(data[0][0].shape[0])
	b = np.random.rand(1)[0]
	acc_counter = 0

	proxy = [i for i in range(0,len(data))]
	random.shuffle(proxy)
	
	for i in range(0, max_iter):
		acc_counter = 0
		for n in proxy:
			x = data[n][0]
			y = data[n][1]

			if y*(np.sum(w*x) + b) <= 0:
				w = w + x*y
				b = b + y
			else:
				acc_counter += 1

		random.shuffle(proxy)

	#print("Training accuracy: " + str(acc_counter/len(data)))

	return w, b


# Load data for training
source_path = sys.argv[1]
nf_data = load_data(source_path + "/negative_polarity/deceptive_from_MTurk")
nr_data = load_data(source_path + "/negative_polarity/truthful_from_Web")
pf_data = load_data(source_path + "/positive_polarity/deceptive_from_MTurk")
pr_data = load_data(source_path + "/positive_polarity/truthful_from_TripAdvisor")

# Count number of docs
nf_num = len(nf_data)
nr_num = len(nr_data)
pf_num = len(pf_data)
pr_num = len(pr_data)
n_docs = nf_num + nr_num + pf_num + pr_num

# Clean & tokenize data 
# Convert into list of dicts for each document
# Overall dict 

nf_local, nf_global = token_sub(nf_data)
nr_local, nr_global = token_sub(nr_data)
pf_local, pf_global = token_sub(pf_data)
pr_local, pr_global = token_sub(pr_data)

# Create a set of words
word_set = set(list(nf_global) + list(nr_global) + list(pf_global) + list(pr_global)) 

# Feature selection  - Top mutual information
# Calculated 2 times - Once per set of classes 

class_a  = [] # Truthful/Deceptive
class_b  = [] # Positive/Negative

for word in word_set:
	nf_count = 0 if word not in nf_global else nf_global[word]
	nr_count = 0 if word not in nr_global else nr_global[word]
	pf_count = 0 if word not in pf_global else pf_global[word]
	pr_count = 0 if word not in pr_global else pr_global[word]

	n = n_docs + 1e-10
	
	# Class a calculation
	n11 = pr_count + nr_count + 1e-10   					 # Truthful and word 
	n01 = pf_count + nf_count + 1e-10  						 # Deceptive and word
	n10 = (pr_num - pr_count) + (nr_num - nr_count) + 1e-10  # Truthful and no word
	n00 = (pf_num - pf_count) + (nf_num - nf_count)	+ 1e-10  # Deceptive and no word

	n1d = pr_num + nr_num + 1e-10   						# Number of truthful documents
	nd1 = n11 + n01 + 1e-10  								# Number of docs with word
	n0d = pf_num + nf_num + 1e-10  							# Number of deceptive documents
	nd0 = n00 + n10	+ 1e-10  								# Number of docs without word 

	mi = (n11/n)*math.log((n*n11)/(n1d*nd1)) + (n01/n)*math.log((n*n01)/(n0d*nd1)) + (n10/n)*math.log((n*n10)/(n1d*nd0)) + (n00/n)*math.log((n*n00)/(n0d*nd0))

	class_a.append((word, mi))
		
	# Class b calculation
	n11 = pr_count + pf_count + 1e-10  						 # Positive and word 
	n01 = nr_count + nf_count + 1e-10  						 # Negative and word
	n10 = (pr_num - pr_count) + (pf_num - pf_count) + 1e-10  # Positive and no word
	n00 = (nr_num - nr_count) + (nf_num - nf_count)	+ 1e-10  # Negative and no word

	n1d = pr_num + pf_num + 1e-10   						# Number of positive documents
	nd1 = n11 + n01 + 1e-10   								# Number of docs with word
	n0d = nr_num + nf_num + 1e-10  							# Number of negative documents
	nd0 = n00 + n10 + 1e-10  								# Number of docs without word 
				
	mi = (n11/n)*math.log((n*n11)/(n1d*nd1)) + (n01/n)*math.log((n*n01)/(n0d*nd1)) + (n10/n)*math.log((n*n10)/(n1d*nd0)) + (n00/n)*math.log((n*n00)/(n0d*nd0))

	class_b.append((word, mi))

# Hyperparam
n_features_a = 1000  # Num tokens for truth/decep net
n_features_b = 1000	 # Num tokens for pos/neg net

# Most useful tokens: Truthful and deceptive
class_a = sorted(class_a, key=lambda x: x[1])[-n_features_a:] 
class_a_words = sorted([i[0] for i in class_a]) # Sort for standardization

# Most useful tokens: Positive and negative
class_b = sorted(class_b, key=lambda x: x[1])[-n_features_b:] 
class_b_words = sorted([i[0] for i in class_b]) # Again, for standardization

# At this point have 2 lists of features

# List of numpy arrays, int tuples
truth_data = make_fun(class_a_words, nr_local + pr_local,1)
decep_data = make_fun(class_a_words, nf_local + pf_local,-1)

# Array of numpy arrays
pos_data = make_fun(class_b_words, pr_local + pf_local,1)
neg_data = make_fun(class_b_words, nr_local + nf_local,-1)


# Train Perceptron

# Hyperparam:
max_iter_vanilla_a = 100
max_iter_avg_a 	   = 100

max_iter_vanilla_b = 100
max_iter_avg_b	   = 100


data_a = truth_data + decep_data

w_vanilla_a, b_vanilla_a = train_vanilla(data_a, max_iter_vanilla_a) # Train truthful deceptive classifier (Vanilla)
w_avg_a, b_avg_a = train_avg(data_a, max_iter_avg_a) # Train turthful deceptive classifier (Avg)


data_b = pos_data + neg_data

w_vanilla_b, b_vanilla_b = train_vanilla(data_b, max_iter_vanilla_b) # Train positive negative classifier (Vanilla)
w_avg_b, b_avg_b = train_avg(data_b, max_iter_avg_b) # Train positive negative classifier (Avg)


# Write vanilla
out_str_vanilla = ""

out_str_vanilla += "{}\n".format(n_features_a)
out_str_vanilla += "{}\n".format(n_features_b)

for i in range(len(class_a_words)):
	out_str_vanilla += "{}|{}\n".format(class_a_words[i],w_vanilla_a[i])
out_str_vanilla += "{}\n".format(b_vanilla_a)

for i in range(len(class_b_words)):
	out_str_vanilla += "{}|{}\n".format(class_b_words[i],w_vanilla_b[i])
out_str_vanilla += "{}".format(b_vanilla_b)

# Write avg
out_str_avg = ""

out_str_avg += "{}\n".format(n_features_a)
out_str_avg += "{}\n".format(n_features_b)

for i in range(len(class_a_words)):
	out_str_avg += "{}|{}\n".format(class_a_words[i],w_avg_a[i])
out_str_avg += "{}\n".format(b_avg_a)

for i in range(len(class_b_words)):
	out_str_avg += "{}|{}\n".format(class_b_words[i],w_avg_b[i])
out_str_avg += "{}".format(b_avg_b)


with open("./vanillamodel.txt", "w") as fp:
	fp.write(out_str_vanilla)

with open("./averagedmodel.txt", "w") as fp:
	fp.write(out_str_avg)
