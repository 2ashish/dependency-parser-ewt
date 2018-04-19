from collections import deque
import numpy as np
from sklearn.preprocessing import CategoricalEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
import math
from keras.utils.np_utils import to_categorical


class Embedding(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		self.dim = 50

	def transform(self, X):
		return np.array([
			np.concatenate([self.word2vec[w] for w in X if w in self.word2vec]
					or [np.zeros(self.dim)], axis=0)
		])

def loadGloveModel(gloveFile):
    #print "Loading Glove Model..."
    f = open(gloveFile,'r')
    embedding_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs 
    f.close()
    print('Found %s word vectors.' % len(embedding_index))
    return embedding_index

def check_child(x, head, beta, sigma):
	ans = 1
	for i in beta:
		# print(i, head[i],x,head[i]-x)
		if(head[i]==x):
			ans = 0
	for i in sigma:
		if(head[i]==x):
			ans = 0
	return ans

def extract_features(words, config):
	def get_stack_context(depth, stack, data):
		if depth >= 3:
			return data[stack[-1]], data[stack[-2]], data[stack[-3]]
		elif depth >= 2:
			return data[stack[-1]], data[stack[-2]], ''
		elif depth == 1:
			return data[stack[-1]], '', ''
		else:
			return '', '', ''

	def get_buffer_context(i, n, data):
		if i + 1 >= n:
			return data[i], '', ''
		elif i + 2 >= n:
			return data[i], data[i + 1], ''
		else:
			return data[i], data[i + 1], data[i + 2]


	features, transition = [], []
	i=0
	for index in config:
		# print('\n\n')
		# print(index[0])
		# i+=1
		sigma = index[0]
		beta = index[1]
		pos = index[2]
		st = index[3]
		
		if beta:
			k = beta[0]
		else:
			k = len(words)-1
		s1,s2,s3 = get_stack_context(len(sigma), sigma, words)
		b1,b2,b3 = get_buffer_context(k, len(words), words)
		features.append([s1,s2,s3,b1,b2,b3,pos])
		transition.append(st)
	return  features,transition



st = open('train.conllu').read().strip().split('\n\n')
feat ,transit = [], []
z=0;
for sent in st:
	print(z)
	z+=1
	lines = [line.split('\t') for line in sent.split('\n') if line[0]!='#']
	words, head, pos, configuration = ['*root'], [' '], [' '], []
	count = 1
	for lin in lines:
		# print(lin)
		if(float(count)!=float(lin[0])):
			continue
		else:
			count+=1
		words.append(lin[2])
		head.append(int(lin[6]))
		pos.append(lin[3])
	#print(head)

	sigma = [0]
	beta = deque([i for i in range(1,len(words))])
	#print(beta)
	#print(head[1])
	#print(check_child(1,head,beta))

	j=0
	while(len(beta)>0) or (sigma != [0]):
	# for i in range(0,100):
		k = len(sigma)
		#print(k,sigma[k-1],sigma[k-2])
		#print(sigma)
		if(k>1):
			if(head[sigma[k-2]]==sigma[k-1]) and (check_child(sigma[k-2],head,beta,sigma)) and (sigma[k-2]!=0):
				configuration.append([list(sigma), list(beta), pos[sigma[k-2]], 1])
				temp = sigma[k-1]; sigma.pop(); sigma.pop(); sigma.append(temp)
				
			else:
				if(head[sigma[k-1]]==sigma[k-2]) and (check_child(sigma[k-1],head,beta,sigma)) and (sigma[k-1]!=0): 
					configuration.append([list(sigma), list(beta), pos[sigma[k-1]], 2])
					sigma.pop()
					
				else:
					if(len(beta)):
						configuration.append([list(sigma), list(beta), pos[beta[0]], 0])
						sigma.append(beta[0])
						beta.popleft()
						

		else:
			if(len(beta)):
				configuration.append([list(sigma), list(beta), pos[beta[0]], 0])
				sigma.append(beta[0])
				beta.popleft()
		
		j+=1
		if (j>100):
			break
	
	# for i in configuration:
	# 	print(i[3])
	features,transition = extract_features(words,configuration)
	for f in features:
		feat.append(f)
	for t in transition:
		transit.append(t)
	# feat.append(features)
	# transit.append(transition)

print(len(feat),25)

GloveDimOption = '50' # this  could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
embeddings_index = loadGloveModel('glove.6B.' + GloveDimOption + 'd.txt') 

# print(embeddings_index['apple'])
# print(embeddings_index['mango'])
embeddings_index['']= np.zeros(50)
embeddings_index['*root']= np.ones(50)

enc = CategoricalEncoder(encoding='onehot')
X_pos = [['ADJ'],['ADP'],['ADV'],['AUX'],['CCONJ'],['DET'],['INTJ'],['NOUN'],['NUM'],['PART'],['PRON'],['PROPN'],['PUNCT'],['SCONJ'],['SYM'],['VERB'],['X']] 
enc.fit(X_pos)

for i in X_pos:
	embeddings_index[i[0]] = pad_sequences(enc.transform([[i[0]]]).toarray(), maxlen=50, padding='post')[0]
	#embeddings_index[i[0]] = pad_sequences(enc.transform([[i[0]]]).toarray(), maxlen=18, padding='post')[0]
	# print(embeddings_index[i[0]])
	# print(embeddings_index['apple'])

feat_vect, transit_vect = [], []
# feat_vect = np.array(())
# transit_vect = np.array(())
for i in feat:
	#print(i)
	sd =np.array(())
	for w in i:
		# if(w in embeddings_index.keys()):
		# 	sd = np.concatenate(sd,embeddings_index[w])
		# else:
		# 	sd = np.concatenate(sd,np.zeros(50))
		try:
			temp = embeddings_index[w]
		except:
			temp = np.zeros(50)

		sd = np.append(sd,temp)

	# sd = np.array([
	# 		np.concatenate([embeddings_index[w] for w in i if w in embeddings_index.keys()]
	# 				or [np.zeros(50)], axis=0)
	# 	])
	#print(sd.shape)
	ind2rem = [i for i in range(318,350)]
	sd = np.delete(sd,ind2rem)
	feat_vect.append(sd)

	#print(len(feat_vect))
feat_vect = np.asarray(feat_vect)
print(feat_vect.shape)
for index in transit:
	transit_vect.append(index)
transit_vect = np.asarray(transit_vect)
print(transit_vect.shape)

# print(len(transit_vect))
# print(len(feat_vect))
k= int(math.floor(0.8*len(transit_vect)))
# print(k)
X_train = feat_vect[0:k]
X_test = feat_vect[k:len(transit_vect)]
Y_train = to_categorical(transit_vect[0:k],num_classes=3)
Y_test = to_categorical(transit_vect[k:len(transit_vect)],num_classes=3)
print(Y_train.shape)
print(X_train.shape)

model = Sequential()
model.add(Dense(100, input_dim=318, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(15, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=32,
          epochs=20,
          validation_data=(X_test, Y_test))