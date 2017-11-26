
# http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/
# https://www.datacamp.com/community/tutorials/machine-learning-python#explore
# Data Manipulation Library  pandas
# 
# digits = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
# training = pd.read_csv("./optdigits.tra", header=None)
# test = pd.read_csv("./optdigits.tes", header=None)

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plot

# digits = datasets.load_digits()

class Study02(object):
    """Read test data, display test data"""

    def __init__(self):
        self.digits = datasets.load_digits()
        print(self.digits.DESCR)
        print(self.digits.data.shape)
        print(np.all(self.digits.images.reshape((1797,64)) == self.digits.data))

    def ShowDigits(self):
        figure = plot.figure(figsize=(6, 6))
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        for i in range(64):
            subplot = figure.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
            subplot.imshow(self.digits.images[i], cmap=plot.cm.binary, interpolation='nearest')
            subplot.text(0, 7, str(self.digits.target[i]))

        plot.show()
    
    def ShowDigits2(self):
        images_and_labels = list(zip(self.digits.images, self.digits.target))
        for index, (image, label) in enumerate(images_and_labels[:8]):
            plot.subplot(2, 4, index + 1)
            
            plot.axis('off')
            
            plot.imshow(image, cmap=plot.cm.gray_r,interpolation='nearest')
            plot.title('Training: ' + str(label))

        plot.show()


c = Study02();
#c.ShowDigits()
c.ShowDigits2()