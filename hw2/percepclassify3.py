

import sys
import os
import string
import glob
import math
import re
import numpy as np

# File-paths 
file_list = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))

# Load classifier
model = "avg"
if "vanilla" in sys.argv[1]:
	model = "vanilla"

with open(sys.argv[1]) as fp:
	model = fp.readlines()

# Parse classifier

n_features_a = int(model[0])
n_features_b = int(model[1])

ordered_words_a = [] 
ordered_words_b = []

w_model_a = []
b_model_a = 0

for i in range(2, 2 + n_features_a):
	word, weight = model[i].split("|")
	ordered_words_a.append(word)
	w_model_a.append(float(weight))
w_model_a = np.array(w_model_a)

b_model_a = float(model[2 + n_features_a])

w_model_b = []
b_model_b = 0

for i in range(2 + n_features_a + 1, len(model) - 1):
	word, weight = model[i].split("|")
	ordered_words_b.append(word)
	w_model_b.append(float(weight))
w_model_b = np.array(w_model_b)

b_model_b = float(model[len(model) - 1])




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
	text_tokens = text.split()

	x_a = np.array([text_tokens.count(word) for word in ordered_words_a])	# Construct class a feature array
	x_b = np.array([text_tokens.count(word) for word in ordered_words_b])	# Consturct class b feature array


	# Return the pos of max 
	class_a = "truthful"
	class_b = "positive"

	if np.sum(x_a*w_model_a) + b_model_a <= 0:
		class_a = "deceptive"

	if np.sum(x_b*w_model_b) + b_model_b <= 0:
		class_b = "negative"
	
	out_str += "{} {} {}\n".format(class_a,class_b,file)	# Add to outstring


# Write out-string to file
out_str = out_str[:-1]

with open("percepoutput.txt","w") as fp:
	fp.write(out_str)
