#!/usr/bin/env python
# coding: utf-8

# # EDA of Project

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
get_ipython().run_line_magic('matplotlib', 'inline')

train_dir = r'E:\DummyData\Project Eye Detection\dataset_new\train' # image folder


OpenEye = [fn for fn in os.listdir(f'{train_dir}\Open') if fn.endswith('.jpg')] #list of open image
ClosedEye= [fn for fn in os.listdir(f'{train_dir}\Closed') if fn.endswith('.jpg')]#list of closed
print( "Open Eyes Image Dataset ",OpenEye)
print( "Closed Eyes Image Dataset",ClosedEye)
     


# # Sample Images of Open & Closed Eyes from Dataset

# In[2]:


select_Open = np.random.choice(OpenEye, 3, replace = False)
select_Close = np.random.choice(ClosedEye, 3, replace = False)

#image matrix 2x3
fig = plt.figure(figsize = (8,6))
for i in range(6):
    if i < 3:
        fp = f'{train_dir}\Open/{select_Open[i]}'
        label = 'Open'
    else:
        fp = f'{train_dir}\Closed/{select_Close[i-3]}'
        label = 'Closed'
    ax = fig.add_subplot(2, 3, i+1)
    
    # to plot without rescaling, remove target_size
    fn = image.load_img(fp, target_size = (100,100), color_mode='grayscale')
    plt.imshow(fn, cmap='Greys_r')
    plt.title(label)
    plt.axis('off')
plt.show()

# also check the number of files here
len(select_Open), len(select_Close)


# # Processing on images data

# In[3]:


def img2np(path, list_of_filename, size = (64, 64)):
    # iterating through each file
    for fn in list_of_filename:
        fp = path + fn
        current_image = image.load_img(fp, target_size = size, 
                                       color_mode = 'grayscale')
        # image to a matrix
        img_ts = image.img_to_array(current_image)
        print("Shape of image:",img_ts.shape)
        # turn that into a vector / 1D array
        img_ts = [img_ts.ravel()]
        print("Converted into 1D array:",img_ts)
        try:
            # concatenate different images
            full_mat = np.concatenate((full_mat, img_ts))
            print("Shape of 1D coverted array:",full_mat.shape)
        except UnboundLocalError: 
            # if not assigned yet, assign one
            full_mat = img_ts
    return full_mat
# run it on our folders
select_OpenEye = img2np(f'{train_dir}\Open/', select_Open)
select_CloseEye = img2np(f'{train_dir}\Closed/', select_Close)


# # Mean of images 

# In[4]:


def find_mean_img(full_mat, title, size = (64, 64)):
    # calculate the average
    mean_img = np.mean(full_mat, axis = 0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(size)
    plt.imshow(mean_img, vmin=0, vmax=255, cmap='Greys_r')
    plt.title(f'Average {title}')
    plt.axis('off')
    plt.show()
    return mean_img

open_mean = find_mean_img(select_OpenEye, 'Open')
close_mean = find_mean_img(select_CloseEye, 'Closed')


# # Difference of Open image mean and closed image mean

# In[5]:


contrast_mean =open_mean - close_mean
plt.imshow(contrast_mean, cmap='bwr')
plt.title(f'Difference Between Open & Close Average')
plt.axis('off')
plt.show()


# # PCA

# In[6]:


from sklearn.decomposition import PCA
from math import ceil

def eigenimages(full_mat, title, n_comp = 0.7, size = (64, 64)):
    # fit PCA to describe n_comp * variability in the class
    pca = PCA(n_components = n_comp, whiten = True)
    pca.fit(full_mat)
    print('Number of PC: ', pca.n_components_)
    return pca
  
def plot_pca(pca, size = (64, 64)):
    # plot eigenimages in a grid
    n = pca.n_components_
    
    fig = plt.figure(figsize=(8, 8))
    r = int(n**0.5)
    c = ceil(n/ r)
    for i in range(n):
        ax = fig.add_subplot(r, c, i + 1, xticks = [], yticks = [])
        ax.imshow(pca.components_[i].reshape(size), 
                  cmap='Greys_r')
    plt.axis('off')
    plt.show()
    
plot_pca(eigenimages(select_OpenEye, 'Open'))
plot_pca(eigenimages(select_CloseEye, 'Closed'))


# In[ ]:





# In[ ]:




