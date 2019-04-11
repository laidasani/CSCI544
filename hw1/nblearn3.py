

# Input:	path to source dir
# Output:	nbmodel.txt

import sys
import os
import string
import math
import re

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

	return text_data, len(text_data)


# Load data for training
source_path = sys.argv[1]
nf_data, nf_num = load_data(source_path + "/negative_polarity/deceptive_from_MTurk")
nr_data, nr_num = load_data(source_path + "/negative_polarity/truthful_from_Web")
pf_data, pf_num = load_data(source_path + "/positive_polarity/deceptive_from_MTurk")
pr_data, pr_num = load_data(source_path + "/positive_polarity/truthful_from_TripAdvisor")

# Count number of docs
n_docs = nf_num + nr_num + pf_num + pr_num

# Clean data

# Create custom tokens
def token_sub(doc_array):
	to_return = []

	map_dict = dict.fromkeys(string.punctuation,"")
	for p in map_dict:
		map_dict[p] = " " + p + " "
	mapping = str.maketrans(map_dict)

	for doc in doc_array:
		new_doc = re.sub(r"[0-1]?[0-9](:\d{0,2}((\s*)?am|(\s*)?pm)?|(\s*)?am|(\s*)?pm)", " timetoken ", doc, flags=re.IGNORECASE) # Identify time
		new_doc = re.sub(r"\$\d*(\.\d?\d)?", " moneytoken ", new_doc) # Identify money
		new_doc = re.sub(r"\d+(\.*)?\d*", " digittoken ", new_doc) # Identify digits
		new_doc = re.sub(r"[A-Z]{1}[A-Z,a-z]*,", " introtoken ", new_doc) # Identify "intro words" - Capital word followed by comma (Ex. Formally, )
		new_doc = new_doc.translate(mapping) # Count punctuation as tokens

		to_return.append(new_doc)

	return to_return


nf_data = token_sub(nf_data)
nr_data = token_sub(nr_data)
pf_data = token_sub(pf_data)
pr_data = token_sub(pr_data)


# Tokenize data - Simple split on space characters
nf_tokens = (" ".join(nf_data).lower()).split()
nr_tokens = (" ".join(nr_data).lower()).split()
pf_tokens = (" ".join(pf_data).lower()).split()
pr_tokens = (" ".join(pr_data).lower()).split()

word_set = set(nf_tokens + nr_tokens + pf_tokens + pr_tokens) 


# Feature selection  - Top mutual information
# Calculated 2 times - Once per set of classes 

class_a  = [] # Truthful/Deceptive
class_b  = [] # Positive/Negative

def count_docs(doc_array, word):
	return len([1 for doc in doc_array if word in doc])

for word in word_set:
	nf_count = count_docs(nf_data,word)
	nr_count = count_docs(nr_data,word)
	pf_count = count_docs(pf_data,word)
	pr_count = count_docs(pr_data,word)

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


# Sort and pick top k tokens (Test w/ k = 100)
class_a = sorted(class_a, key=lambda x: x[1])[-1000:] # Truthful and deceptive
class_a_words = [i[0] for i in class_a]

class_b = sorted(class_b, key=lambda x: x[1])[-4000:] # Positive and negative
class_b_words = [i[0] for i in class_b]

truth_tokens = [tok for tok in pr_tokens if tok in class_a_words] + [tok for tok in nr_tokens if tok in class_a_words]
decep_tokens = [tok for tok in pf_tokens if tok in class_a_words] + [tok for tok in nf_tokens if tok in class_a_words]

# Train Naive Bayes

# Train truthful deceptive classifier
train_dict_a = {}

for word in class_a_words:
	truth_p = (1 + truth_tokens.count(word))/(len(truth_tokens) + len(class_a))
	decep_p = (1 + decep_tokens.count(word))/(len(decep_tokens) + len(class_a))

	train_dict_a[word] = [truth_p, decep_p]

# Train positive negative classifier
train_dict_b = {}

pos_tokens = [tok for tok in pr_tokens if tok in class_b_words] + [tok for tok in pf_tokens if tok in class_b_words]
neg_tokens = [tok for tok in nr_tokens if tok in class_b_words] + [tok for tok in nf_tokens if tok in class_b_words]

for word in class_b_words:
	pos_p = (1 + pos_tokens.count(word))/(len(pos_tokens) + len(class_b))
	neg_p = (1 + neg_tokens.count(word))/(len(neg_tokens) + len(class_b))

	train_dict_b[word] = [pos_p, neg_p]


# Write to file
out_str = ""

out_str += "{}\n{}\n{}\n{}\n".format((pr_num+nr_num)/n_docs, (pf_num+nf_num)/n_docs, (pr_num + pf_num)/n_docs, (nr_num + nf_num)/n_docs)

for word in train_dict_a:
	out_str += "a {} {} {}\n".format(word,train_dict_a[word][0],train_dict_a[word][1])

for word in train_dict_b:
	out_str += "b {} {} {}\n".format(word,train_dict_b[word][0],train_dict_b[word][1])

out_str = out_str[:-1]

with open("./nbmodel.txt","w") as fp:
	fp.write(out_str)


