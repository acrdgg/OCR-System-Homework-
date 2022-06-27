from typing_extensions import final
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from multiprocessing import Pool
import os
from train import Features
from train import image_map
from train import image_namelist
from train import map_count

Features_t = []
count = 0
chara_count = {}

#read test img
img = io.imread('test.bmp')

#binarize
th = 206
img_binary = (img<th).astype(np.double)

#find each characters
img_label = label(img_binary, background=0)
regions = regionprops(img_label)
ax = plt.gca()

for props in regions:
    minr, minc, maxr, maxc = props.bbox
    if(np.abs(maxc-minc)<10 or np.abs(maxr-minr)<10):
        continue
    roi = img_binary[minr:maxr, minc:maxc];
    m = moments(roi);
    cc = m[0,1] / m[0,0];
    cr = m[1,0] / m[0,0];
    mu = moments_central(roi, center=(cr,cc));
    nu = moments_normalized(mu);
    hu = moments_hu(nu);
    Features_t.append(hu);
    ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1));

ax.set_title('Bounding boxes')
io.imshow(img)
io.show()


D = cdist(Features_t, Features)
#io.imshow(D)
D_index = np.argsort(D, axis=1)
plt.title('Distance Matrix')

print(image_map)
print(image_namelist)
print(D_index)
# print(D_index.shape)
# print(D_index.shape[0])

#initialize
ini_count = 0
while ini_count < map_count:
    chara_count[ini_count] = 0
    ini_count += 1

print("recognized characters in test ", D_index.shape[0])
#recog - standard
# rec_count = 0

# while rec_count < D_index.shape[0]:
#     temppos = 0
#     while D_index[rec_count][0] > image_map[temppos]:
#         temppos += 1
#     chara_count[temppos] += 1
#     rec_count += 1

#recog - enhancememt
rec_count = 0
enhance_output = {}
while rec_count < D_index.shape[0]:
    hori_count = 0
    #initialize temp hashmap
    ini2_count = 0
    while ini2_count < map_count:
        enhance_output[ini2_count] = 0
        ini2_count += 1
    #start counting
    while hori_count < 49:
        temppos = 0
        while D_index[rec_count][hori_count] > image_map[temppos]:
            temppos += 1
        enhance_output[temppos] += 1
        hori_count += 1
    finalpos = max(enhance_output, key=enhance_output.get)
    chara_count[finalpos] += 1
    rec_count += 1


#finalize
final_count = 0
outputlist = {}
while final_count < map_count:
    outputlist[image_namelist[final_count]] = chara_count[final_count]
    final_count += 1


print(outputlist)

#io.show()
outputlist



