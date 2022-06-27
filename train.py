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


Features=[]
path1 = "/home/acrdgg/Desktop/cs334_hw4/images/";

listing = os.listdir(path1);
#print(listing);
count = 0;
listcount = 0;
map_count = 0;
image_map = {};
image_namelist = {};


for file in listing:
    img = io.imread(path1+file);
    
    
    #print(img.shape);

    #io.imshow(img);
    #plt.title('original image');
    #io.show();

    # hist = exposure.histogram(img);
    # plt.bar(hist[1], hist[0]);
    # plt.title('histogram');
    # plt.show();

    th = 206
    img_binary = (img < th).astype(np.double);
    # io.imshow(img_binary);
    # plt.title('binary image');
    # io.show();

    img_label = label(img_binary, background = 0);
    # io.imshow(img_label);
    # plt.title('labeled image');
    # io.show();
    #print(np.amax(img_label));



    regions = regionprops(img_label);
    #io.imshow(img_binary);
    ax = plt.gca();
    
    fine_boxes = 0;
    hu_numbers = 0;
    for props in regions:
            minr, minc, maxr, maxc = props.bbox;
            if (np.abs(maxc-minc) < 10):
                continue
            roi = img_binary[minr:maxr, minc:maxc];
            m = moments(roi);
            cc = m[0,1] / m[0,0];
            cr = m[1,0] / m[0,0];
            mu = moments_central(roi, center=(cr,cc));
            nu = moments_normalized(mu);
            hu = moments_hu(nu);
            Features.append(hu);
            hu_numbers += 1;
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1));
            fine_boxes += 1;

    count += fine_boxes;
    image_map[map_count] = count;
    image_namelist[map_count] = file;
    map_count += 1;
    # print("image:", file, "stop at", count, " fine boxes number: ", fine_boxes);
    # print("current features numbers: ", len(Features));
    ax.set_title('Bounding boxes');
    io.imshow(img)
    io.show()

#print(image_map);
# print("Features's Length: ",len(Features));

mean_std = {};
i = 0;

while i < len(Features):
    temp_mean = np.mean(Features[i]);
    temp_std = np.std(Features[i]);
    

    Features[i]-=temp_mean;
    Features[i]/=temp_std;
    
    mean_std[i] = [temp_mean,np.var(Features[i])];
    i += 1;
    
#print(Features);
#print(image_map)
#print(image_namelist)
#print(map_count)

Features



#distance maxtrix calculation
# D = cdist(Features,Features);
# D_index = np.argsort(D, axis=1);
# io.imshow(D);
# plt.title('Distance Matrix');
# print(D)
#np.set_printoptions(threshold=np.inf)
#print(D_index);
#print(D_index.shape)
# io.show();






