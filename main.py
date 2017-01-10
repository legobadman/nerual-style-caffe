import caffe
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy

# load original image p, init image x
content_img_org = caffe.io.load_image('Neckarfront.jpg')
style_img_org = caffe.io.load_image('starry_night_google.jpg')

# load vgg network model
VGGweights = 'vgg_normalised.caffemodel'
VGGmodel = 'VGG_ave_pool_deploy.prototxt'

caffe.set_mode_cpu()
#caffe.set_device(0)
mean = np.array([ 0.40760392,  0.45795686,  0.48501961])
im_size = np.array([256., 256.], dtype='int64')
net_mean = np.tile(mean[:,None,None],(1,) + tuple(im_size.astype(int)))
net = caffe.Classifier(VGGmodel, VGGweights, mean=net_mean, channel_swap=(2,1,0), input_scale=255,)
content_img = net.transformer.preprocess('data', content_img_org)[None,:]
style_img = net.transformer.preprocess('data', style_img_org)[None,:]

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


style_layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']
style_layers = ['conv1_1', 'pool1', 'conv2_1', 'pool2', 'conv3_1', 'pool3', 'pool4', 'pool5']

layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

#style_layers = ['conv1_1', 'pool5']
#style_layers = ['conv1_1','pool4']
G = []
W = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9]
alpha = 1
beta = 1000
A = []

net.forward(data=np.reshape(style_img, style_img.shape))
for layer in layers:
    G.append(G_matrix(net.blobs[layer].data))

net.forward(data=np.reshape(content_img, content_img.shape))
for layer in layers:
    A.append(net.blobs[layer].data.copy())


def f_style(x):
    net.forward(data=np.reshape(x, source_img.shape))
    loss = 0.

    grad = np.zeros(net.blobs['data'].diff.shape)
    for layer in net.blobs:
        net.blobs[layer].diff[...] = np.zeros_like(net.blobs[layer].diff)

    for i in range(len(style_layers)-1,-1,-1):
        layer = style_layers[i]
        activations = net.blobs[layer].data.copy()
        N = activations.shape[1]
        M  = np.prod(np.array(activations.shape[2:]))
        F = activations.reshape(N, -1)
        G_ = np.dot(F, F.T) / M
    
        F_ = net.blobs[layer].data.reshape(N, -1).copy()

        E = float(W[i])/4 * (np.square(G_ - G[i]).sum()) / N**2
        loss += E

        f_grad = float(W[i]) * ((F_.T).dot(G_ - G[i])).T / (M * N**2)
        #f_grad[f_grad < 0] = 0
        net.blobs[layer].diff[:] += f_grad.reshape(activations.shape)
        net.blobs[layer].diff[(net.blobs[layer].data == 0)] = 0.
        #grad = net.backward(start='conv1_1')['data'].copy()
        if i > 0:
            net.backward(start=style_layers[i], end=style_layers[i-1])
        else:
            grad = net.backward(start=layer)['data'].copy()

    print loss
    return [loss, np.array(grad.ravel(), dtype=float)]

def f_content(x):
    net.forward(data=np.reshape(x, source_img.shape))
    loss = 0.
    grad = np.zeros(net.blobs['data'].diff.shape)
    for layer in net.blobs:
        net.blobs[layer].diff[...] = np.zeros_like(net.blobs[layer].diff)

    for i in range(len(content_layers)-1,-1,-1):
        layer = content_layers[i]
        activations = net.blobs[layer].data.copy()
        loss += 0.5 * np.sum(np.square(activations - A[i]))
        net.blobs[layer].diff[:] += (activations - A[i]).copy()
        if i > 0:
            net.backward(start=content_layers[i], end=content_layers[i-1])
        else:
            grad = net.backward(start=layer)['data'].copy()
    
    print loss
    return [loss, np.array(grad.ravel(), dtype=float)]

def style_content(x):
    def style_gradient(activatinos):
        N = activations.shape[1]
        M  = np.prod(np.array(activations.shape[2:]))
        F = activations.reshape(N, -1)
        G_ = np.dot(F, F.T) / M
        F_ = net.blobs[layer].data.reshape(N, -1).copy()
        loss = float(W[i])/4 * (np.square(G_ - G[i]).sum()) / N**2
        f_grad = float(W[i]) * ((F_.T).dot(G_ - G[i])).T / (M * N**2)
        return loss, f_grad.reshape(activations.shape)

    def content_gradient(activations):
        loss = 0.5 * np.sum(np.square(activations - A[i]))
        f_grad = (activations - A[i]).copy()
        return loss, f_grad

    net.forward(data=np.reshape(x, content_img.shape))
    loss = 0.

    grad = np.zeros(net.blobs['data'].diff.shape)
    for layer in net.blobs:
        net.blobs[layer].diff[...] = np.zeros_like(net.blobs[layer].diff)

    for i in range(len(layers)-1,-1,-1):
        layer = layers[i]
        activations = net.blobs[layer].data.copy()
        style_loss, style_grad = style_gradient(activations)
        content_loss, content_grad = content_gradient(activations)
        loss += alpha * style_loss + beta * content_loss
        net.blobs[layer].diff[:] += alpha * style_grad + beta * content_grad
        if i > 0:
            net.backward(start=layers[i], end=layers[i-1])
        else:
            grad = net.backward(start=layer)['data'].copy()
        
    print loss
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
bounds = get_bounds([content_img], im_size)
minimize_options={'maxiter': 10, 'maxcor': 20, 'ftol': 0, 'gtol': 0}

def random_count(x):
    print np.random.random()

result = minimize(style_content, init, method='L-BFGS-B', jac=True, bounds = bounds, options=minimize_options)#, )
print result

new_texture = result['x'].reshape(*content_img.shape[1:]).transpose(1,2,0)[:,:,::-1]
new_texture = histogram_matching(new_texture, content_img_org)
plt.imshow(new_texture)
plt.show()
# plt.imshow(source_img_org)
# plt.show()
# only conv 1




# optimize from the gradient, and get the result image

# show the result image
