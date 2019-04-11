
import sys
import math

# Use train path to load training data line by line
with open(sys.argv[1]) as fp:
	train_data = fp.readlines()


emission_dict = {} # Emission prob, P(word|tag)

transition_dict = {} # Transition prob,  P(n_tag|c_tag)

for data in train_data:

	# Split each line into tokens
	tokens = data.split()

	prev_tag = "ST@RT" # Handle start tag 
	
	for elmt in tokens: # Don't handle end tag

		word, tag = elmt.rsplit("/", 1) # Split on last "/"
		
		# Previous tag has never been seen before
		if prev_tag not in transition_dict:		
			transition_dict[prev_tag] = {}

		# Init/update transition freq
		transition_dict[prev_tag][tag] = 1 if tag not in transition_dict[prev_tag] else transition_dict[prev_tag][tag] + 1

		if tag not in emission_dict:
			emission_dict[tag] = {}

		# Init/update emission freq
		emission_dict[tag][word] = 1 if word not in emission_dict[tag] else emission_dict[tag][word] + 1

		prev_tag = tag

	# Handle end
	if prev_tag not in transition_dict:
		transition_dict[prev_tag] = {}


	transition_dict[prev_tag]["3ND"] = 1 if "3ND" not in transition_dict[prev_tag] else transition_dict[prev_tag]["3ND"] + 1


# Make prob and take log of prob
tag_list = list(transition_dict.keys()) + ["3ND"]

for tag in transition_dict: 
	
	# Add 1 smoothing
	total_events = 0
	for n_tag in tag_list:
		if n_tag not in transition_dict[tag]:
			transition_dict[tag][n_tag] = 1
		else:
			transition_dict[tag][n_tag] += 1
		
		total_events += transition_dict[tag][n_tag]

	# Log prob
	for n_tag in transition_dict[tag]:
		transition_dict[tag][n_tag] = math.log(transition_dict[tag][n_tag]/total_events)

for tag in emission_dict:

	total_events = 0
	for word in emission_dict[tag]:
		total_events += emission_dict[tag][word]
	
	for word in emission_dict[tag]:
		emission_dict[tag][word] = math.log(emission_dict[tag][word]/total_events)


# Collect as string
out_str = ""

for tag in transition_dict:
	out_str += "{} {}".format(tag, "T")
	for n_tag in transition_dict[tag]:
		out_str += " {}/{} ".format(n_tag,transition_dict[tag][n_tag])
	out_str += "\n"

for tag in emission_dict:
	out_str += "{} {}".format(tag, "E")
	for word in emission_dict[tag]:
		out_str += " {}/{}".format(word,emission_dict[tag][word])
	out_str += "\n"

out_str = out_str[:-1]

# Save everything to file
with open("./hmmmodel.txt", "w") as fp:
	fp.write(out_str)