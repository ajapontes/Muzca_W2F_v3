import numpy as np

data = np.load('./data/mel_fwod_dataset_variable.npz', allow_pickle=True)
mel = data['mel']

formas = [x.shape for x in mel]
from collections import Counter
print(Counter(formas))