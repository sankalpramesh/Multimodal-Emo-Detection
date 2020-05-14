import pickle
import numpy as np 
from sklearn.decomposition import PCA
import pandas as pd 

train = pickle.load(open('train_video_emb.pkl', 'rb'), encoding='latin1')
dev = pickle.load(open('dev_video_emb.pkl', 'rb'), encoding='latin1')
test = pickle.load(open('test_video_emb.pkl', 'rb'), encoding='latin1')

df_train = pd.read_csv('../../MELD.Raw/train_sent_emo.csv')
df_dev = pd.read_csv('../../MELD.Raw/dev_sent_emo.csv')
df_test = pd.read_csv('../../MELD.Raw/test_sent_emo.csv')

max_uttr = 33
pca = PCA(n_components=200)

train_new = {}
dev_new = {}
test_new = {}

# train
all_data_arr = []
count_arr = {}
dialogue_id = 0
s = 0
while(dialogue_id < 1039):
	count = len(df_train[df_train['Dialogue_ID'] == dialogue_id])
	count_arr[dialogue_id] = count

	if str(dialogue_id) in train:
		s += count
		utterance = train[str(dialogue_id)]

		for j in range(count):
			all_data_arr.append(train[str(dialogue_id)][j])

	dialogue_id += 1

print(s)
all_data_arr = np.array(all_data_arr)
print(all_data_arr.shape)

reduced_data = pca.fit_transform(all_data_arr)
print(reduced_data.shape)


for key, value in train.items():
	count = count_arr[int(key)]
	arr = reduced_data[int(key): int(key)+count]
	new_arr = []
	if(len(arr) < max_uttr):
		for i in range(max_uttr):
			if i < count:
				new_arr.append(arr[i])
			else:
				new_arr.append([0.]*200)
	else:
		new_arr = arr

	train_new[int(key)] = np.array(new_arr)
print(len(train_new))





# dev
all_data_arr = []
count_arr = {}
dialogue_id = 0
while(dialogue_id < 114):
	count = len(df_dev[df_dev['Dialogue_ID'] == dialogue_id])
	count_arr[dialogue_id] = count

	if str(dialogue_id) in dev:
		utterance = dev[str(dialogue_id)]

		for j in range(count):
			all_data_arr.append(dev[str(dialogue_id)][j])

	dialogue_id += 1

all_data_arr = np.array(all_data_arr)
print(all_data_arr.shape)

reduced_data = pca.fit_transform(all_data_arr)
print(reduced_data.shape)

for key, value in dev.items():
	count = count_arr[int(key)]
	arr = reduced_data[int(key): int(key)+count]
	new_arr = []
	if(len(arr) < max_uttr):
		for i in range(max_uttr):
			if i < count:
				new_arr.append(arr[i])
			else:
				new_arr.append([0.]*200)
	else:
		new_arr = arr

	dev_new[int(key)] = np.array(new_arr)
print(len(dev_new))



# test
all_data_arr = []
count_arr = {}
dialogue_id = 0
while(dialogue_id < 280):
	count = len(df_test[df_test['Dialogue_ID'] == dialogue_id])
	count_arr[dialogue_id] = count

	if str(dialogue_id) in test:
		utterance = test[str(dialogue_id)]

		for j in range(count):
			all_data_arr.append(test[str(dialogue_id)][j])

	dialogue_id += 1

all_data_arr = np.array(all_data_arr)
print(all_data_arr.shape)

reduced_data = pca.fit_transform(all_data_arr)
print(reduced_data.shape)

for key, value in test.items():
	count = count_arr[int(key)]
	arr = reduced_data[int(key): int(key)+count]
	new_arr = []
	if(len(arr) < max_uttr):
		for i in range(max_uttr):
			if i < count:
				new_arr.append(arr[i])
			else:
				new_arr.append([0.]*200)
	else:
		new_arr = arr

	test_new[int(key)] = np.array(new_arr)
print(len(test_new))


pickle.dump(train_new, open('train_video_emb_reduced.pkl', 'wb'))
pickle.dump(dev_new, open('val_video_emb_reduced.pkl', 'wb'))
pickle.dump(test_new, open('test_video_emb_reduced.pkl', 'wb'))