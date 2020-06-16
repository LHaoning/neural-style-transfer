import os
import sys
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import pprint
from nst_utils import *
import imageio


# %matplotlib inline

# pp = pprint.PrettyPrinter(indent=4)
# model = load_vgg_model("pre-trained-model/imagenet-vgg-verydeep-19.mat")


# pp.pprint(model)


# @ Computing the content cost
# FUNCTION: compute_content_cost
def compute_content_cost(a_C, a_G):
    '''

    :param a_C -- the tensor of shape (1, H, W, C), hidden layer activations representing content of the image C
    :param a_G -- the tensor of shape (1, H, W, C), hidden layer activations representing content of the image G
    :return: J_content -- the content cost between image G and image C
    '''

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, shape=[m, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, -1, n_C])

    # Compute the cost
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

    return J_content


# @ Computing the style cost
# FUNCTION: gram_matrix
def gram_matrix(A):
    '''

    :param A: matrix of shape (C, H*W)
    :return: Gram matrix of A, of shape (C, C)
    '''
    GA = tf.matmul(A, tf.transpose(A))

    return GA


# FUNCTION: compute_layer_style_cost
def compute_layer_style_cost(a_S, a_G):
    '''

    :param a_S: tensor of shape (1, H, W, C), hidden layer activations representing style of the image S
    :param a_G: tensor of shape (1, H, W, C), hidden layer activations representing style of the image G
    :return: J_style_layer: tensor representing style cost of given layer between G and S
    '''

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape images to shape (n_C, n_H*n_W)
    a_S = tf.reshape(tf.transpose(a_S, perm=[0, 3, 1, 2]), shape=[n_C, -1])
    a_G = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), shape=[n_C, -1])

    # Computing gram_matrix for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * (n_H * n_W) * (n_H * n_W) * (n_C * n_C))

    return J_style_layer


# Style weights for different layers
STYLE_LAYERS = [
    ('conv1_1', .2),
    ('conv2_1', .2),
    ('conv3_1', .2),
    ('conv4_1', .2),
    ('conv5_1', .2)
]


# FUNCTION: compute_style_cost
def compute_style_cost(model, STYLE_LAYERS):
    '''

    :param model: the loaded model
    :param STYLE_LAYERS: the weights for style cost of different layers
    :return: style cost
    '''

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


# @ Defining the total cost to optimize
# FUNCTION: total_cost
def total_cost(J_content, J_style, alpha=10, beta=40):
    '''

    :param J_content: the content cost between G and C
    :param J_style: the style cost between G and S
    :param alpha: hyperparam. weighting the importance of the content cost
    :param beta: hyperparam. weighting the importance of the style cost
    :return: J: total cost
    '''

    J = alpha * J_content + beta * J_style

    return J


# @ Solve the optimization problem
# Reset the graph
# tf.reset_default_graph()
# Start interactive session
sess = tf.Session()

# Content image
content_image = imageio.imread('images/louvre_small.jpg')
content_image = reshape_and_normalize_image(content_image)

# Style image
style_image = imageio.imread('images/monet.jpg')
style_image = reshape_and_normalize_image(style_image)

# Initialize generated image correlated with content image
generated_image = generate_noise_image(content_image)
# imshow(generated_image[0])
# plt.show()

# Load pre-trained model
model = load_vgg_model("pre-trained-model/imagenet-vgg-verydeep-19.mat")

# content cost
# Assign the content image to be the input of the model
sess.run(model['input'].assign(content_image))
# Select the output tensor of layer conv4_2
out = model['conv4_2']
# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)
# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out
# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# style cost
# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))
# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

# total cost
J = total_cost(J_content, J_style, alpha=10, beta=40)

# define optimizer
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step
train_step = optimizer.minimize(J)

# FUNCTION: model
def model_nn(sess, input_image, num_iterations=200):
    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image through the model
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image

model_nn(sess, generated_image)

