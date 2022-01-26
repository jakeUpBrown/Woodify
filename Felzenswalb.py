#importing needed libraries
import skimage.segmentation
from matplotlib import pyplot as plt
import math

#reading image
zionPic = "Zion 1.jpg"
girlPic = "Girl Face 1.jpg"
img = plt.imread(girlPic)

#performing segmentation
#first for k=50
#seconf for k=1000

fig = plt.figure(figsize=(10, 7))

kStart = 50
kInc = 50
kMax = 500
numIter = int((kMax - kStart) / kInc) + 1

k = kStart
iter = 0
imgList = [None] * numIter

cols = 4
rows = math.ceil(numIter / cols)

while(k <= kMax):
    imgList[iter] = skimage.segmentation.felzenszwalb(img, scale=k)

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, cols, iter + 1)

    # showing image
    plt.imshow(imgList[iter])
    plt.axis('off')
    plt.title('k=' + str(k))

    k += kInc
    iter += 1

plt.show()
#res1 = skimage.segmentation.felzenszwalb(img, scale=50)
#res2 = skimage.segmentation.felzenszwalb(img, scale=1000)

#printing the results
#fig = plt.figure(figsize=(12, 5))
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
#ax1.imshow(res1); ax1.set_xlabel("k=50")
#ax2.imshow(res2); ax2.set_xlabel("k=1000")

#fig.suptitle("Graph based image segmentation")
#plt.tight_layout()