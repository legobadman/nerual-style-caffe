import caffe
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import scipy
import os

from preprocess import *

# load original image p, init image x
content_name = "miao.jpg"
content_img_org = caffe.io.load_image('content/' + content_name)
style_name = "night.jpg"
style_img_org = caffe.io.load_image('style/' + style_name)

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


content_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
content_layers = ['conv1_1']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
style_layers = []

layers = list(set(content_layers) | set(style_layers))
layers = sorted(layers,lambda layer1, layer2 : net.blobs.keys().index(layer1) < net.blobs.keys().index(layer2))
layers.sort()

W = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9]
#W = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
alpha = 10
beta = 1
A = {}



net.forward(data=np.reshape(style_img, style_img.shape))
G = {}
for layer in style_layers:
    G[layer] = G_matrix(net.blobs[layer].data)

net.forward(data=np.reshape(content_img, content_img.shape))
A = {}
W_styles = {"conv1_1":1e9, "conv2_1":1e9, "conv3_1":1e9, "conv4_1":1e9, "conv5_1":1e9}

for layer in content_layers:
    A[layer] = net.blobs[layer].data.copy()


loss = 0.

def style_content(x):
    def style_gradient(layer):
        activations = net.blobs[layer].data.copy()
        target_G = G[layer]
        N = activations.shape[1]
        M  = np.prod(np.array(activations.shape[2:]))
        F = activations.reshape(N, -1)
        G_ = np.dot(F, F.T) / M
        F_ = net.blobs[layer].data.reshape(N, -1).copy()
        loss = float(W_styles[layer]) / 4 * (np.square(G_ - target_G).sum()) / N**2
        f_grad = float(W_styles[layer]) * ((F_.T).dot(G_ - target_G)).T / (M * N**2)
        return loss, f_grad.reshape(activations.shape)

    def content_gradient(layer):
        activations = net.blobs[layer].data.copy()
        target_A = A[layer]
        loss = 0.5 * np.sum(np.square(activations - target_A))
        f_grad = (activations - target_A).copy()
        return loss, f_grad

    global loss

    net.forward(data=np.reshape(x, content_img.shape))
    loss = 0.

    grad = np.zeros(net.blobs['data'].diff.shape)
    for layer in net.blobs:
        net.blobs[layer].diff[...] = np.zeros_like(net.blobs[layer].diff)

    for i in range(len(layers)-1,-1,-1):
        layer = layers[i]
        if not layer in style_layers and not layer in content_layers:
            continue

        if layer in style_layers:
            style_loss, style_grad = style_gradient(layer)
            loss += beta * style_loss
            net.blobs[layer].diff[:] += beta * style_grad

        if layer in content_layers:
            content_loss, content_grad = content_gradient(layer)
            loss += alpha * content_loss
            net.blobs[layer].diff[:] += alpha * content_grad

        if i > 0:
            net.backward(start=layers[i], end=layers[i-1])
        else:
            grad = net.backward(start=layer)['data'].copy()

    return [loss, np.array(grad.ravel(), dtype=float)]


bounds = get_bounds([content_img], im_size)


filename = ""
if content_layers != []:
    filename += "content(%s) %s, " % (content_name, ' '.join(content_layers))
if style_layers != []:
    filename += "style(%s) %s " % (style_name, ' '.join(style_layers))

dir_name = filename


iters = 0
def show_iter(x, net, title=None, handle=False, show_img=True, save_img=False):
    global iters
    print 'iters = %d, loss = %d' % (iters, loss)
    
    if iters == 0:
        if dir_name not in os.listdir('.'):
            os.mkdir(dir_name)
        os.chdir(dir_name)
    elif iters % 10 == 0:
        if show_img:
            disp_image = (x.reshape(*net.blobs['data'].data.shape)[0].transpose(1,2,0)[:,:,::-1]-x.min())/(x.max()-x.min())
            clear_output()
            plt.imshow(disp_image)
            if title != None:
                ax = plt.gca()
                ax.set_title(title)
            f = plt.gcf()
            display()
            plt.show()
        if save_img:
            save_result(x, net, "%s.png" % str(iters))

    iters += 1
    if handle:
        return f

minimize_options={'maxiter': 100, 'maxcor': 20, 'ftol': 0, 'gtol': 0}
#result = minimize(style_content, init, method='L-BFGS-B', jac=True, bounds = bounds, callback=lambda x: show_progress(x, net), options=minimize_options)
result = minimize(style_content, init, method='L-BFGS-B', jac=True, bounds = bounds, 
                    callback=lambda x: show_iter(x, net, show_img=False, save_img=True), 
                    options=minimize_options)
#result = minimize(style_content, init, method='L-BFGS-B', jac=True, bounds = bounds, options=minimize_options)



print result