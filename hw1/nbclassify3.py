
# Input: Path to test data
# Output: nboutput.txt


import sys
import os
import string
import glob
import math
import re

# File-paths 
file_list = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

# Load NB classifier data
with open("./nbmodel.txt") as fp:
	model_data = fp.readlines()

class_prob = [float(p) for p in model_data[:4]]
cond_prob = model_data[4:]

cond_dict_a = {}
cond_dict_b = {}

for line in cond_prob:
	c, word, class_0p, class_1p  = line.split(" ")

	if c == "a":
		cond_dict_a[word] = [float(class_0p), float(class_1p)]
	else:
		cond_dict_b[word] = [float(class_0p), float(class_1p)]

out_str = ""
for file in file_list:

	# Load data into string
	with open(file) as fp:
		text = "".join(fp.readlines())

	# Clean 
	map_dict = dict.fromkeys(string.punctuation,"")
	for p in map_dict:
		map_dict[p] = " " + p + " "

	mapping = str.maketrans(map_dict)

	text = re.sub(r"[0-1]?[0-9](:\d{0,2}((\s*)?am|(\s*)?pm)?|(\s*)?am|(\s*)?pm)", " timetoken ", text, flags=re.IGNORECASE) # Identify time
	text = re.sub(r"\$\d*(\.\d?\d)?", " moneytoken ", text) # Identify money
	text = re.sub(r"\d+(\.*)?\d*", " digittoken ", text) # Identify digits
	text = re.sub(r"[A-Z]{1}[A-Z,a-z]*,", " introtoken ", text) # Identify "intro words" - Capital word followed by comma (Ex. Formally, )
	text = text.translate(mapping) # Count punctuation as tokens
	text = text.lower()
	
	# Tokenize
	text_tokens = text.split()

	truth_prob = math.log(class_prob[0])
	decep_prob = math.log(class_prob[1])
	pos_prob = math.log(class_prob[2])
	neg_prob = math.log(class_prob[3])
	
	# Apply NB concurrently for each word
	for word in text_tokens:

		# Ignore words not in training sets
		
		if word in cond_dict_a: 
			truth_prob += math.log(cond_dict_a[word][0])
			decep_prob += math.log(cond_dict_a[word][1])

		if word in cond_dict_b: 
			pos_prob += math.log(cond_dict_b[word][0])
			neg_prob += math.log(cond_dict_b[word][1])

	# Return the pos of max 
	class_a = "truthful"
	class_b = "negative"

	if truth_prob < decep_prob:
		class_a = "deceptive"

	if neg_prob < pos_prob:
		class_b = "positive"
	
	out_str += "{} {} {}\n".format(class_a,class_b,file)	# Add to outstring


# Write out-string to file
out_str = out_str[:-1]

with open("nboutput.txt","w") as fp:
	fp.write(out_str)
