# %%
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle

f = open('/home/knishi-rmp/github/cifar10/data/cifar-10-batches-py/data_batch_1', 'rb')
datadict = cPickle.load(f,encoding='latin1')
f.close()
X = datadict["data"]
Y = datadict['labels']
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

# %%
index = 46
fig = plt.imshow(X[index:(index+1)][0])
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('image.png', bbox_inches='tight', pad_inches = 0)
