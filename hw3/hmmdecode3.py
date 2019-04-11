
import sys

# Use train path to load training data line by line
with open("./hmmmodel.txt") as fp:
	model = fp.readlines()

# Load model param into data structures
transition_dict = {}
emission_dict = {}

# Every line represents either tag-tag or tag-word information
for elmnts in model:
	
	raw = elmnts.split()
	tag, param_type = raw[0], raw[1]
	param = raw[2:]

	if tag not in transition_dict:
		transition_dict[tag] = {}
	
	if tag not in emission_dict:
		emission_dict[tag] = {}


	# Have to seperate into transition and emission structs
	for pieces in param:
		word_or_tag, log_prob = pieces.rsplit("/",1)

		# Make positive so can take max and add to respective struct
		if param_type == "T": 
			transition_dict[tag][word_or_tag] = float(log_prob)  

		else:
			emission_dict[tag][word_or_tag] = float(log_prob) 

# Helper function
def get_most_prob(current_state, path_prob, transition_dict):
	back_pointer = ""
	max_prob = -sys.maxsize

	for state in path_prob:

		if max_prob < path_prob[state]+transition_dict[state][current_state]:
			max_prob =  path_prob[state]+transition_dict[state][current_state]
			back_pointer = state

	return max_prob, back_pointer




# Want a set of all tags
tag_space  = set(list(transition_dict.keys()))
# This inst a real tag
tag_space.remove("ST@RT")

# Want a list of all words
word_space = set([word for tag in list(emission_dict.keys()) for word in list(emission_dict[tag].keys())])

# Viterbi algorithm 
def label_sentence(line):
	words = line.split()

	# Observer first state
	max_prob = {} 
	back_pointer = {}

	max_prob[0] = {}
	back_pointer[0] = {}

	for state in tag_space:

		# Only want to save states which result in non-zero prob
		# Because tag transition is smoothed - 0 emission is only way for this to happen
		if words[0] in word_space and words[0] in emission_dict[state]: 
			max_prob[0][state] = transition_dict["ST@RT"][state]+emission_dict[state][words[0]]
			back_pointer[0][state] = "ST@RT"

		# If word isn't present in corpus then just consider transition prob
		elif words[0] not in word_space:
			max_prob[0][state] = transition_dict["ST@RT"][state]
			back_pointer[0][state] = "ST@RT"
	
	# Go through the graph 
	for index in range(1,len(words)):
		max_prob[index] = {}
		back_pointer[index] = {}

		for state in tag_space:

			# Only want to save states which result in non-zero prob
			# Because tag transition is smoothed - 0 emission is only way for this to happen
			if words[index] in word_space and words[index] in emission_dict[state]: 
				# Get path prob to current state
				max_prob[index][state], back_pointer[index][state] = get_most_prob(state, max_prob[index-1], transition_dict)
				
				max_prob[index][state] += emission_dict[state][words[index]] # Multiply by emission prob

			# If word isn't present in corpus then just consider transition prob
			elif words[index] not in word_space:
				# Get path prob to current state
				max_prob[index][state], back_pointer[index][state] = get_most_prob(state, max_prob[index-1], transition_dict)

	# Back-prob to find the start of the best path
	end_tag  = "" 
	end_prob = -sys.maxsize

	for state in max_prob[len(words) -1]:
		
		# Factor in transition prob from final tag to "3ND" tag
		if max_prob[len(words)-1][state] + transition_dict[state]["3ND"] > end_prob:
			end_tag = state
			end_prob = max_prob[len(words)-1][state] + transition_dict[state]["3ND"]


	# Backtrack and join with words
	tagged_line = []
	predicted = end_tag
	for i in range(len(words) - 1, -1, -1):
		tagged_line.append("{}/{}".format(words[i], predicted))
		predicted = back_pointer[i][predicted]

	return " ".join(tagged_line[::-1]) 

with open(sys.argv[1]) as fp:
	to_label = fp.readlines()


out_str = ""
for line in to_label:
	out_str += label_sentence(line) + "\n"

out_str = out_str[:-1]

# Save everything to file
with open("./hmmoutput.txt", "w") as fp:
	fp.write(out_str)