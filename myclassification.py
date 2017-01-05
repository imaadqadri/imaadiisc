from time import time
t0=time()
print "importing......"
from time import time
from skimage import color
import matplotlib.pyplot as plt
from loader import  hsvrgb_images
import numpy as np

from nolearn.dbn import DBN


data,labels=hsvrgb_images()
print "training data shape is:",data.shape
print "label shape is",labels.shape
# defining DBN model
dbn = DBN([data.shape[1],1000,1800,400, 3],learn_rates = 0.002,learn_rate_decays = 0.9,output_act_funct="Softmax",epochs = 100,verbose = 1)

print "activating ...."
dbn.fit(data,labels)   #fitted classifier on whole prepared data
print "fitting done!"
# now importing test image
a=(plt.imread("test_data/test2cropped.jpg").astype('float32'))/255.0
# g= a/255.0
h=a.reshape(-1,3)
# pic=plt.imread("test_data/test2cropped.jpg")
hhh=plt.imread("test_data/test2cropped.jpg")
f=color.rgb2hsv(hhh).reshape(-1,3)

test_image=np.concatenate((h,f),axis=1)
print "test image shape:",test_image.shape
print test_image

z=plt.imread("test_data/test2cropped.jpg")
l,b,d=tuple(z.shape)
# new_image_reshaped=new_image.reshape(-1,3)

#predicting on test image
print"predicting on test image....."
# print b.shape , "heeeeeeeeeeeeeee"
pred=dbn.predict(test_image)
# print"prediction shape:",pred.shape
preds=pred.reshape(l,b)
print "predictions changed shape:",preds.shape

#recreate new image
new_image_of_array=np.zeros((l,b,d))
for i in xrange(l):
    for j in xrange(b):
        if preds[i][j]==1:
            new_image_of_array[i][j]=[0,0,0]
        elif preds[i][j]==2:
            new_image_of_array[i][j]=[255,0,0]

        else:
            new_image_of_array[i][j]=[0,255,0]

print "new image generated!"
print "time taken:",round(time()-t0,4)

plt.figure(1)
plt.clf()
plt.imshow(z)
plt.figure(2)
plt.clf()
plt.imshow(new_image_of_array/255)
plt.show()







