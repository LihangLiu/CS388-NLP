import glob
from random import shuffle

MAX_LENGTH = 100

COMMON_PREFIX = ['anti', 'de', 'dis', 'en', 'em', 'fore', 'in', 'im', 'il', 'ir',
				'inter', 'mid', 'mis', 'non', 'over', 'pre', 're', 'semi', 'sub', 
				'super', 'trans', 'un', 'under']
COMMON_SUFFIX = ['able', 'ible', 'al', 'ial', 'ed', 'en', 'er', 'est', 'ful', 'ic',
				'ing', 'ion', 'tion', 'ation', 'ition', 'ity', 'ty', 'ive', 'ative',
				'itive', 'less', 'ly', 'ment', 'ness', 'ous', 'eous', 'ious', 'es',
				's', 'y']

def if_contains_a_common_prefix(word, common_prefix):
	feature = []
	word = word.lower()
	for prefix in common_prefix:
		if word.startswith(prefix):
			feature.append(True)
		else:
			feature.append(False)
	return feature

def if_contains_a_common_suffix(word, common_suffix):
	feature = []
	word = word.lower()
	for suffix in common_suffix:
		if word.endswith(suffix):
			feature.append(True)
		else:
			feature.append(False)
	return feature

class PreprocessData:

	def __init__(self, dataset_type='wsj'):
		self.vocabulary = {}
		self.pos_tags = {}
		self.dataset_type = dataset_type

	## Get standard split for WSJ
	def get_standard_split(self, files):
		if self.dataset_type == 'wsj':
			train_files = []
			val_files = []
			test_files = []
			for file_ in files:
				partition = int(file_.split('/')[-2])
				if partition >= 0 and partition <= 18:
					train_files.append(file_)
				elif partition <= 21:
					val_files.append(file_)
				else:
					test_files.append(file_)
			return train_files, val_files, test_files
		else:
			raise Exception('Standard Split not Implemented for '+ self.dataset_type)

	@staticmethod
	def isFeasibleStartingCharacter(c):
		unfeasibleChars = '[]@\n'
		return not(c in unfeasibleChars)

	@staticmethod
	def get_orthographic_features(w):
		# if capicalized, contains a hyphen, starts with a number
		feature = [w[0].isupper(), ('-' in w), w[0].isdigit()]
		# if contains a common prefix
		feature += if_contains_a_common_prefix(w, COMMON_PREFIX)
		# if contains a common suffix
		feature += if_contains_a_common_suffix(w, COMMON_SUFFIX)
		return [int(f) for f in feature]

	## unknown words represented by len(vocab)
	def get_unk_id(self, dic):
		return len(dic)

	def get_pad_id(self, dic):
		return len(self.vocabulary) + 1

	## get id of given token(pos) from dictionary dic.
	## if not in dic, extend the dic if in train mode
	## else use representation for unknown token
	def get_id(self, pos, dic, mode):
		if pos not in dic:
			if mode == 'train':
				dic[pos] = len(dic)
			else:
				return self.get_unk_id(dic)
		return dic[pos]

	## Process single file to get raw data matrix
	def processSingleFile(self, inFileName, mode):
		matrix = []
		row = []
		with open(inFileName) as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				if line == '':
					pass
				else:
					tokens = line.split()
					for token in tokens:
						## ==== indicates start of new example					
						if token[0] == '=':
							if row:
								matrix.append(row)
							row = []
							break
						elif PreprocessData.isFeasibleStartingCharacter(token[0]):
							wordPosPair = token.split('/')
							if len(wordPosPair) == 2:
								## get ids for word and pos tag
								feature = self.get_id(wordPosPair[0], self.vocabulary, mode)
								tag_id = self.get_id(wordPosPair[1], self.pos_tags, 'train')
								# get orthographic features
								orth_features = PreprocessData.get_orthographic_features(wordPosPair[0])
								row.append((feature, orth_features, tag_id))
		if row:
			matrix.append(row)
		return matrix 			# (num_lines, seq_length) of tuple (word_id, orth_features, tag_id)


	## get all data files in given subdirectories of given directory
	def preProcessDirectory(self, inDirectoryName, subDirNames=['*']):
		if not(subDirNames):
			files = glob.glob(inDirectoryName+'/*.pos')
		else:
			files = [glob.glob(inDirectoryName+ '/' + subDirName + '/*.pos')
					for subDirName in subDirNames]
			files = set().union(*files)
		return list(files)


	## Get basic data matrix with (possibly) variable sized senteces, without padding
	def get_raw_data(self, files, mode):
		matrix = []
		for f in files:
			matrix.extend(self.processSingleFile(f, mode))
		return matrix

	def split_data(self, data, fraction):
		split_index = int(fraction*len(data))
		left_split = data[:split_index]
		right_split = data[split_index:]
		if not(left_split):
			raise Exception('Fraction too small')
		if not(right_split):
			raise Exception('Fraction too big')
		return left_split, right_split

	## Get rid of sentences greater than max_size
	## and pad the remaining if less than max_size
	def get_processed_data(self, mat, max_size):
		X = [] 		# (num_sentences, seq_length)
		y = []		# (num_sentences, seq_length)
		orth_features = []		# (num_sentences, seq_length, num_orth_feature)
		original_len = len(mat)
		mat = filter(lambda x: len(x) <= max_size, mat)
		no_removed = original_len - len(mat)
		for row in mat:
			X_row = [tup[0] for tup in row]
			orth_features_row = [tup[1] for tup in row]
			y_row = [tup[2] for tup in row]
			## padded words represented by len(vocab) + 1
			X_row = X_row + [self.get_pad_id(self.vocabulary)]*(max_size - len(X_row))
			## Padded orth features represented by [0]
			orth_features_row = orth_features_row + [[0]*len(orth_features_row[0])]*(max_size - len(orth_features_row))
			## Padded pos tags represented by -1
			y_row = y_row + [-1]*(max_size-len(y_row))
			X.append(X_row)
			orth_features.append(orth_features_row)
			y.append(y_row)

		return X, orth_features, y, no_removed
