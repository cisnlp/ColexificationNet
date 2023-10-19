import gensim
import sys
import pandas as pd
import os

# obtain all distinct language's ISO 639-3 codes
def get_langs():
	csv_path = '/mounts/Users/student/yihong/Documents/concept_align'
	pbc_info = pd.read_csv(f"{csv_path}/pbc_table.csv", converters={"language_code": str})
	langs = pbc_info['language_code'].values
	langs = sorted(list(set(langs)))
	return langs

def read_pbc_script_data():
	csv_path = '/mounts/Users/student/yihong/Documents/concept_align'
	pbc_info = pd.read_csv(f"{csv_path}/pbc_table.csv", converters={"language_code": str})
	dict_script = dict(zip(pbc_info['file_name'], pbc_info['script_code']))
	return dict_script

def find_version_most(language_code):
	if language_code == 'eng':
		return '/nfs/datc/pbc/eng-x-bible-literal.txt', 'eng-x-bible-literal.txt'
	versions = []
	for filename in os.listdir('/nfs/datc/pbc/'):
		fp = os.path.join('/nfs/datc/pbc/', filename)
		if os.path.isfile(fp) and filename[:3] == language_code:
			versions.append(fp)
	if len(versions) == 0:
		raise ValueError
	elif len(versions) == 1:
		parts = versions[0].split('/')
		return versions[0], parts[-1]
	else:
		# check if there is a newworld version for this language:
		newworld = []
		candidate = ('', 0)
		for v in versions:
			if 'newworld' in v:
				newworld.append(v)
			f = open(v, 'r', encoding="utf-8")
			length_f = len(f.readlines())
			f.close()
			if length_f > candidate[1]:
				candidate = (v, length_f)
		if len(newworld) == 0:
			parts = candidate[0].split('/')
			return candidate[0], parts[-1]
		elif len(newworld) == 1:
			return newworld[0], newworld[0].split('/')[-1]
		else:
			# multiple new world is available
			for v in newworld:
				f = open(v, 'r', encoding="utf-8")
				length_f = len(f.readlines())
				f.close()
				if length_f > candidate[1]:
					candidate = (v, length_f)
			return candidate[0], candidate[0].split('/')[-1]


def read_verses(path):
	contents = []
	verseIDs = []
	with open(path, 'r', encoding="utf-8") as f:
		for line in f.readlines():
			if line[0] == "#":
				continue
			parts = line.strip().split('\t')
			if len(parts) == 2:
				verseIDs.append(parts[0])
				contents.append(parts[1])
	# creating a dictionary for efficiency
	contents_dict = dict(zip(verseIDs, contents))
	return verseIDs, contents_dict


class PBCSentences(object):
	def __init__(self, langs: list):
		self.filenames = []
		self.file_langs = dict()
		for lang in langs:
			path, filename = find_version_most(lang)
			self.filenames.append(path)
			self.file_langs[path] = lang
 
	def __iter__(self):
		print(f"One epoch finished")
		for fname in self.filenames:
			_, ids_contents = read_verses(fname)
			for verse_id, verse_string in ids_contents.items():
				tokens = verse_string.split()
                # here we use the lowercase of each token
				t_sentences = [[verse_id, f'{self.file_langs[fname]}:{x.lower()}'] for x in tokens]
				for item in t_sentences:
					yield item


all_langs = get_langs()
sentences = PBCSentences(all_langs)

epochs = 50
emb_dim = 200
min_count = 2

print(f"Training epochs: {epochs}")
model = gensim.models.Word2Vec(sentences=sentences, vector_size=emb_dim, min_count=min_count, workers=10, sg=1, epochs=epochs)
model.save(f"./word2vec_{epochs}_{emb_dim}.model")
print("Done")


"""

export PYTHONIOENCODING=utf8; nohup python -u train_sentence_ID_script.py > ./sentence_ID_train_log.txt 2>&1 &
server: delta
pid: 43887

"""


