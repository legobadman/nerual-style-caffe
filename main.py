import caffe
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy

# load original image p, init image x
source_img_org = caffe.io.load_image('pebbles.jpg')

# load vgg network model
VGGweights = 'vgg_normalised.caffemodel'
VGGmodel = 'VGG_ave_pool_deploy.prototxt'

caffe.set_mode_cpu()
#caffe.set_device(0)
mean = np.array([ 0.40760392,  0.45795686,  0.48501961])
im_size = np.array([256., 256.], dtype='int64')
net_mean = np.tile(mean[:,None,None],(1,) + tuple(im_size.astype(int)))
net = caffe.Classifier(VGGmodel, VGGweights, mean=net_mean, channel_swap=(2,1,0), input_scale=255,)
source_img = net.transformer.preprocess('data', source_img_org)[None,:]

ndim = 256
init = np.random.randn(*net.blobs['data'].data.shape)

N1 = 64
M1 = 256

def G_matrix(FeatureMap):
    N = FeatureMap.shape[1]
    F = FeatureMap.reshape(N, -1)
    M = F.shape[1]
    G = np.dot(F,F.T) / M
    return G

def get_bounds(images, im_size):
    '''
    Helper function to get optimisation bounds from source image.

    :param images: a list of images
    :param im_size: image size (height, width) for the generated image
    :return: list of bounds on each pixel for the optimisation
    '''

    lowerbound = np.min([im.min() for im in images])
    upperbound = np.max([im.max() for im in images])
    bounds = list()
    for b in range(im_size[0]*im_size[1] * 3):
        bounds.append((lowerbound,upperbound))
    return bounds

layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']
#layers = ['pool4']
G = []
W = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9]

net.forward(data=np.reshape(source_img, source_img.shape))
for layer in layers:
    G.append(G_matrix(net.blobs[layer].data))

def f_style(x):
    net.forward(data=np.reshape(x, source_img.shape))
    loss = 0.

    grad = np.zeros(net.blobs['data'].diff.shape)
    for i, layer in enumerate(layers):
        activations = net.blobs[layer].data.copy()
        N = activations.shape[1]
        M  = np.prod(np.array(activations.shape[2:]))
        F = activations.reshape(N, -1)
        G_ = np.dot(F, F.T) / M
    
        F_ = net.blobs[layer].data.reshape(N, -1).copy()

        E = float(W[i])/4 * (np.square(G_ - G[i]).sum()) / M**2
        loss += E

        f_grad = float(W[i]) * ((F_.T).dot(G_ - G[i])).T / ((M * N)**2)
        #f_grad[f_grad < 0] = 0
        net.blobs[layer].diff[:] = f_grad.reshape(activations.shape)
        net.blobs[layer].diff[(net.blobs[layer].data == 0)] = 0.
        #grad = net.backward(start='conv1_1')['data'].copy()
        net.backward(start=layer)
        grad = grad + net.blobs['data'].diff.copy()

    print loss
    return [loss, np.array(grad.ravel(), dtype=float)]

def f_content(x):
    net.forward(data=np.reshape(x, (1, 3, ndim, ndim)), end='conv1_1')
    F = net.blobs['conv1_1'].data.copy()
    loss = 0.5 * np.sum(np.square(F - P))
    F_grad = (F - P)
    net.blobs['conv1_1'].diff[:] = F_grad
    net.backward(start='conv1_1')
    grad = net.blobs['data'].data.copy()
    return [loss, np.array(grad.ravel(), dtype=float)]

def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram

    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))

def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''

    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r>cum_values.max()] = cum_values.max()
        matched_image = inv_cdf(r).reshape(org_image.shape)
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
            r[r>cum_values.max()] = cum_values.max()
            matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)

    return matched_image


#loss, f_grad = L_content(init)
bounds = get_bounds([source_img], im_size)
minimize_options={'maxiter': 50, 'maxcor': 20, 'ftol': 0, 'gtol': 0}

def random_count(x):
    print np.random.random()

result = minimize(f_style, init, method='L-BFGS-B', jac=True, bounds = None, options=minimize_options)#, )
print result

new_texture = result['x'].reshape(*source_img.shape[1:]).transpose(1,2,0)[:,:,::-1]
new_texture = histogram_matching(new_texture, source_img_org)
plt.imshow(new_texture)
plt.show()
# plt.imshow(source_img_org)
# plt.show()
# only conv 1




# optimize from the gradient, and get the result image

# show the result image
