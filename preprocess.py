import numpy as np

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

def show_progress(x, net, title=None, handle=False):
    '''
    Helper function to show intermediate results during the gradient descent.

    :param x: vectorised image on which the gradient descent is performed
    :param net: caffe.Classifier object defining the network
    :param title: optional title of figuer
    :param handle: obtional return of figure handle
    :return: figure handle (optional)
    '''

    disp_image = (x.reshape(*net.blobs['data'].data.shape)[0].transpose(1,2,0)[:,:,::-1]-x.min())/(x.max()-x.min())
    clear_output()
    plt.imshow(disp_image)
    if title != None:
        ax = plt.gca()
        ax.set_title(title)
    f = plt.gcf()
    display()
    plt.show()
    if handle:
        return f

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
