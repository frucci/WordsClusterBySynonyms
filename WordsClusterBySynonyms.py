import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy import sparse
import nltk
from nltk.corpus import wordnet as wn
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud

class WordsClusterBySynonyms:
	""" 
	WordClusterBySynonyms will be able to generate clusters from a dataframe of words. Parameters:
	
	dataframe = input dataframe
	name = name of the input columns
	
	lang = languages 
	n_jobs = number of workers
	
	"""
	def __init__(self, dataframe, name,  lang='ita', n_jobs=-1):
		self.lang = lang
		self.dataframe = dataframe
		self.name = name
		self.n_jobs = n_jobs
		self._name_sin = None
		self._idx_name = None
		self._name_idx = None
		self.result = None
		self.final = None
		self.final_enabler = 0
		
	def _create_mydistance(self, criteria=min):
		def mydistance(w1, w2, b):
			app_x = b[w1]
			app_y = b[w2]
			intersect = len(set(app_x) & set(app_y))
			try:
				x = intersect / criteria(len(app_x), len(app_y))
				return (x)
			except ZeroDivisionError:
				return (0)
		return mydistance
	
	def _symmetrize(self, a):
		return a + a.T - np.eye(a.shape[0])
	
	######################################################################################################
	
	def get_synonym(self, w):
		return (list({
			s
			for ss in wn.synsets(w, lang=self.lang)
			for s in ss.lemma_names(lang=self.lang)
		}))
	
	
	def get_synonyms_pandas(self, dataframe = None, threshold = None):
		if dataframe is None:
			dataframe = self.dataframe
		dataframe['synonym'] = dataframe[self.name].apply(lambda x: self.get_synonym(x))
		dataframe['len_syn'] = dataframe.apply(axis=1, func = lambda x: len(x.synonym))
		
		drop = dataframe[dataframe['len_syn'] == 0].shape[0]
		if drop > 0:
			print("There are {} words with no synonyms".format(drop))
			print("I've deleted all of them")
		
		dataframe = dataframe[dataframe['len_syn'] > 0].reset_index(drop=True)
		
		if threshold!= None:
			dataframe = dataframe[dataframe['len_syn'] < threshold].reset_index(drop=True)
		
		dataframe = dataframe.drop('len_syn', axis=1)
		
		# _create_group_dict
		self._name_sin = dataframe.set_index(self.name).to_dict()['synonym']
		self._idx_name = dataframe.to_dict()[self.name]
		self._name_idx = {v: k for k, v in self._idx_name.items()}
		self.dataframe = dataframe
		
		return dataframe
	
	def plot_hist(self, save=False, name=None, dataframe = None):
		if dataframe is None:
			dataframe = self.dataframe
		dataframe['len_syn'] = dataframe.apply(axis=1, func = lambda x: len(x.synonym))
		dataframe = dataframe[dataframe['len_syn'] > 0].reset_index(drop=True)
		plt.figure(figsize=(15,3))
		plt.title('Histogram of the number of synonyms')
		plt.hist(dataframe['len_syn'].values, bins=100)
		if save == True:
			plt.savefig(name+'.jpg')
		plt.show()
		
	
	def set_threshold(self, threshold, dataframe = None):
		if dataframe is None:
			dataframe = self.dataframe
		dataframe['len_syn'] = dataframe.apply(axis=1, func = lambda x: len(x.synonym))
		dataframe = dataframe[dataframe['len_syn'] < threshold].reset_index(drop=True).copy()
		dataframe.drop('len_syn', axis=1, inplace=True)
		# _create_group_dict
		self._name_sin = dataframe.set_index(self.name).to_dict()['synonym']
		self._idx_name = dataframe.to_dict()[self.name]
		self._name_idx = {v: k for k, v in self._idx_name.items()}
		
		self.dataframe = dataframe
		return dataframe
		
	def create_distance_matrix(self,                               
							mydistance=None, criteria=None,
							verbose=False):
		"""
		
		"""
		if criteria == None:
			criteria = min
		
		if mydistance == None:
			mydistance = self._create_mydistance(criteria=criteria)
			
			
		start_time = time()
		app = 0
		row, col, data = [], [], []
		tot = len(self._idx_name)
		for i in range(0, tot):
			for j in range(i, tot):
				distance = mydistance(self._idx_name[i], self._idx_name[j], self._name_sin)
				if distance != 0:
					row.append(i)
					col.append(j)
					data.append(distance)
			app += 1
	
			if verbose == True:
				"Computation is started..."
				if i == 100:
					print('First ' + str(i) + ' worked in ' +
						str(round((time() - start_time) / 60, 2)) +
						' minutes')
					print('I should finish in ' + str(
						round((time() - start_time) / 60 * len(self._name_sin) /
							100, 2)) + ' minutes')
	
		matrixd = self._symmetrize(
			sparse.csr_matrix((data, (row, col))).todense())
		print("Computation ended")
		return 1 - matrixd
	
	def run_cluster(self, eps, min_samples, distance_matrix):
		db = DBSCAN(
			eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=self.n_jobs).fit(distance_matrix)
		self.result = db
		return db
	
	def plot_eps_ncluster(self, distance_matrix, start=0.01, stop=1, ntot=25, min_samples=5, save=False, name=None):
		x = np.linspace(start, stop, ntot)
		res = [self.run_cluster(eps, min_samples, distance_matrix) for eps in x]
		
		y = np.array(
			[len(set(res[i].labels_)) for i in range(0,len(res))])
		y2 = np.array(
			[(res[i].labels_ == -1).sum() for i in range(0,len(res))])
		
		plt.figure(figsize=(15,3))
		plt.plot(x, y)
		plt.xlabel('eps')
		plt.ylabel('Number of clusters')
		if save == True:
			plt.savefig(name+'_clusters.jpg')
		plt.show()
		
		plt.figure(figsize=(15,3))
		plt.plot(x, y2, color='red')
		plt.xlabel('eps')
		plt.ylabel('Number of words  not clustered')
		if save == True:
			plt.savefig(name+'_not_clustered.jpg')
		plt.show()
		
	def get_labeled_pandas(self, dataframe=None, drop = True, reset_index=False):
		if dataframe is None:
			dataframe = self.dataframe
		temp = self.get_synonyms_pandas()
		if self.result==None:
			print("run run_cluster before then get_labeled_pandas")
			return 			
		dataframe = pd.concat([temp ,pd.DataFrame(self.result.labels_)],axis=1)
		dataframe.columns = list(temp.columns) + ['label']
		if drop == True:
			dataframe = dataframe[dataframe['label']!=-1]
			if reset_index==True:
				dataframe.reset_index(drop=True, inplace=True)
		self.final = dataframe
		self.final_enabler = 1
		return self.final
		
		
	def plot_cluster_k(self, distance_matrix, word, background_color='white', save=False, name=None):
		if self.final_enabler == 0:
			print("Please run get_labeled_pandas before then plot_cluster_k")
			return
		try:
			x = self.final['label'][self.final[self.name]==word]
			index_x = x.index.values[0]
			x = x.values[0]
		except IndexError:
			print("I'm sorry. {} not in dataframe".format(word))
			return
			
		df = self.final[self.final['label'] == x][self.name]
		cloud = list(set(df) - set([word]))
	
		indexes = [self.final[self.final[self.name]==w].index.values[0] for w in cloud]
		
		wordcloud = {}
		for i in indexes:
			wordcloud[" " + self._idx_name[i] + " "] = int((1 - distance_matrix[index_x,i] + 2))
			
		text = [key * val for key, val in wordcloud.items()]    
		text = " ".join(text).strip()
		
		wordc = WordCloud(background_color= background_color , max_words=len(set(text.split())))
		wordc.generate(text)
		
		plt.figure(figsize=(15,5))
		plt.imshow(wordc, interpolation='bilinear')
		plt.axis("off")
		if save == True:
			plt.savefig(name+'.jpg')
		plt.show()