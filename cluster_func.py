import numpy as np
from skimage import io
from sklearn.cluster import KMeans, SpectralClustering, Birch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

### this could be so much better!
def cluster_image(image_path, clusters, alg, savename='', **kwargs) : 
    original = io.imread(image_path)
    n_colors = clusters
    arr = original.reshape((-1, 3))
    clstr = alg(n_clusters=n_colors, **kwargs).fit(arr)
    labels = clstr.labels_
    centers = clstr.cluster_centers_
    less_colors = centers[labels].reshape(original.shape).astype('uint8')
    #plt.figure(figsize=(40,40),facecolor='white')
    #io.imshow(less_colors)
    io.imsave(savename, less_colors)
    
def cluster_image_spacefeatures(image_path, clusters, alg, savename='', **kwargs) : 
    original = io.imread(image_path)
    rng = original.shape[0:2]
    x,y = np.meshgrid(range(rng[0]), range(rng[1]))
    scl = MinMaxScaler()
    coordinate = np.concatenate([original.reshape(-1,3), x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    n_colors = clusters
    clstr = alg(n_clusters=n_colors, **kwargs).fit(scl.fit_transform(coordinate))
    labels = clstr.labels_
    centers = clstr.cluster_centers_
    unscal = scl.inverse_transform(centers[labels])
    less_colors = unscal.reshape(tuple(list(original.shape[0:2]) +[5])).astype('uint8')[:,:,0:3]
    #plt.figure(figsize=(40,40),facecolor='white')
    #io.imshow(less_colors)
    io.imsave(savename, less_colors)

def cluster_image2(image, clusters, alg, **kwargs) : 
    n_colors = clusters
    original = io.imread(image
                         )
    arr = original.reshape((-1,3))
    clstr = alg(n_clusers=n_colors, **kwargs).fit(arr)
    labels = clstr.labels_
    centers = clstr.cluster_centers_
    less_colors = centers[labels].reshape(original.shape).astype('uint8')
    return less_colors