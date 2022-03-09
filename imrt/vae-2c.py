# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 20:31:47 2022

@author: jper0011
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from os.path import exists
from contextlib import redirect_stdout
from scipy.spatial.distance import cdist



# hyperparameter dictionary for gridsearch
e=30           # 'epochs'
bs=32           # 'batch_size'
lr=0.0001         # 'learn_rate'
ld=2            # 'latent_dim'
k1=3            # 'kernel size layer 1'
k2=3            # 'kernel size layer 2'
k3=3            # 'kernel size layer 3'
n1=32           # 'num filters layer 1'
n2=64           # 'num filters layer 2'
n3=128           # 'num filters layer 3'
bn=1            # 'batch_norm' yes/no
rls=10000        # 'reconstruction_loss_scalar'
nd=32           # 'dense_layer_size




# get data
data = np.load(r'x_dta_dd.npy').astype("float32")
labels = np.load(r'y_dta_dd.npy')

# get input shape
img_height = data.shape[1]
img_width = data.shape[2]
num_channels = data.shape[3]
input_shape = (img_height, img_width, num_channels)  # 'input_shape'

# shuffle data
labels, data = shuffle(labels, data, random_state=2)

# separate train/test
split = len(data)//10
x_test = data[:split]
x_train = data[split:]
y_test = labels[:split]
y_train = labels[split:]





# check filenames
lossname0 = 'loss_history'
latentname0 = 'latent_space'
hypername0 = 'hyperparameters'
encodername0 = 'vae_encoder'
decodername0 = 'vae_decoder'
fcmname0 = 'fuzzy_means'
file_exists = exists(latentname0 + '.png')
if not file_exists:
    latentname = latentname0 + '.png'
    lossname = lossname0
    hypername = hypername0 + '.txt'
    encodername = encodername0 + '.h5'
    decodername = decodername0 + '.h5'
    fcmname = fcmname0 + '.png'
else:
    i=0
    while file_exists:
        i += 1
        latentname = latentname0 + '_' + str(i) + '.png'
        lossname = lossname0 + '_' + str(i)
        hypername = hypername0 + '_' + str(i) + '.txt'
        encodername = encodername0 + '_' + str(i) + '.h5'
        decodername = decodername0 + '_' + str(i) + '.h5'
        fcmname = fcmname0 + '_' + str(i) + '.png'

        file_exists = exists(latentname)





# define loss history potting function
def plothistory(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['reconstruction_loss'])
    plt.plot(history.history['kl_loss'])
    plt.title(f'model loss, {e} epochs')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'recon loss', 'kl loss'], loc='upper left')
    plt.savefig(lossname + '.png')
    




# sampler for reparameterization
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=2)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Define the VAE as a Model with a custom train_step
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = 10000*K.mean(K.square(data-reconstruction), axis=[1, 2, 3])
            
            # change reconstruction loss from binary cross entropy to mse
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #     )
            # )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }





# model goes in here #

# Build the encoder
latent_dim = ld  #  latent dimension 
encoder_inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(n1, k1, strides=1, padding="same")(encoder_inputs)  #  added 5x5 layer w/ zero stride (doesn't change dimension)
x = layers.LeakyReLU(alpha=0.1)(x)
if bn == 1:
    x = tf.keras.layers.BatchNormalization()(x)
x = layers.Conv2D(n2, k2, strides=2, padding="same")(x)
x = layers.LeakyReLU(alpha=0.1)(x)
if bn == 1:
    x = tf.keras.layers.BatchNormalization()(x)
x = layers.Conv2D(n3, k3, strides=2, padding="same")(x)
x = layers.LeakyReLU(alpha=0.1)(x)
if bn == 1:
    x = tf.keras.layers.BatchNormalization()(x)
conv_shape = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(nd, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# Build the decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation="relu")(latent_inputs)
x = layers.Reshape((conv_shape[1],conv_shape[2],conv_shape[3]))(x)
x = layers.Conv2DTranspose(n3, k3, strides=2, padding='same', output_padding=1)(x)  # needed padding of 1 to match dimension
x = layers.LeakyReLU(alpha=0.1)(x)
if bn == 1:
    x = tf.keras.layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(n2, k2, strides=2, padding="same", output_padding=1)(x)  # needed padding of 1 to match dimension
x = layers.LeakyReLU(alpha=0.1)(x)
if bn == 1:
    x = tf.keras.layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(n1, k1, strides=1, padding="same")(x)  #  same 5x5 layer to mimic the encoder (dimension is already set)
x = layers.LeakyReLU(alpha=0.1)(x)
if bn == 1:
    x = tf.keras.layers.BatchNormalization()(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="tanh", padding="same")(x)  #  changed activation function to 'tanh' instead of 'sigmoid'
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# # load weights:
# e_model = 'vae_encoder.h5'
# d_model = 'vae_decoder.h5'
# encoder = keras.models.load_model(e_model)
# decoder = keras.models.load_model(d_model)





# compile and train vae
vae = VAE(encoder, decoder)  # define vae as connection between encoder and decoder
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))  # compile the model with adam optimizer
history = vae.fit(x_train, epochs=e, batch_size=bs, verbose=1)  # train and save loss history




# plot loss history
plothistory(history)





# # new error type not used for training:
# x_e1 = np.load(r'xerror1.npy').astype("float32")
# y_e1 = np.load(r'yerror1.npy')
# mu3, sigma3, z3 = encoder.predict(x_e1)





# save image of latent space
plt.figure(figsize=(15,15))
#ax = plt.axes(projection='3d')
ax = plt.axes()

mu1, sigma1, z1 = encoder.predict(x_train)      # training data
# pa = plt.scatter(mu1[:, 0], mu1[:, 1], mu1[:,2], marker='.', c=y_train)
pa = plt.scatter(mu1[:, 0], mu1[:, 1], marker='.', c=y_train)

mu2, sigma2, z2 = encoder.predict(x_test)       # test data
#pb = plt.scatter(mu2[:, 0], mu2[:, 1], mu2[:,2], marker='x', c=y_test)
pb = plt.scatter(mu2[:, 0], mu2[:, 1], marker='x', c=y_test)

plt.xlabel('mu0')
plt.ylabel('mu1')
plt.title(f'latent space of training data, {e} epoch')

plt.legend()
#plt.colorbar(pa, label='training data')
#plt.colorbar(pb, label='test data')
#ax.set_facecolor('xkcd:black')

plt.savefig(latentname)








# save trained model to file
vae.encoder.save(encodername)
vae.decoder.save(decodername)

# print hyperparameters to file
with open(hypername, 'w') as f:
    f.write('VAE HYPERPARAMETERS:\n')
    f.write(f'--epochs={e}\n')
    f.write(f'--batch_size={bs}\n')
    f.write(f'--learning_rate={lr}\n')
    f.write(f'--latent_dim={ld}\n')
    f.write(f'--batch_norm={bn}\n')
    f.write(f'--dense_layer_size={nd}\n')
    f.write(f'ENCODER FILTERS:\n')
    f.write(f'--Conv1: size=({k1}x{k1}), number={n1}\n')
    f.write(f'--Conv2: size=({k2}x{k2}), number={n2}\n')
    f.write(f'--Conv3: size=({k3}x{k3}), number={n3}\n')
    f.write(f'DECODER FILTERS:\n')
    f.write(f'--ConvT1: size=({k3}x{k3}), number={n3}\n')
    f.write(f'--ConvT2: size=({k2}x{k2}), number={n2}\n')
    f.write(f'--ConvT3: size=({k1}x{k1}), number={n1}\n')
    f.write(f'--ConvT4 (Output): size=(3x3), number=1\n')
    f.write('\n')
    with redirect_stdout(f):
        encoder.summary()
    with redirect_stdout(f):
        decoder.summary()


def _cmeans0(data, u_old, c, m):
    """
    Single step in generic fuzzy c-means clustering algorithm.
    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.
    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / (np.ones((data.shape[1],
                                    1)).dot(np.atleast_2d(um.sum(axis=1))).T)

    d = _distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d

def _distance(data, centers):
    """
    Euclidean distance from each point to each cluster center.
    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.
    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.
    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers).T

def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.
    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.
    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.
    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)

def cmeans(data, c, m=2, error=1e-3, maxiter=300, init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].
    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.
    Returns
    -------
    cntr : 2d array, size (S, c)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.
    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.
    Fuzzy C-Means has a known problem with high dimensionality datasets, where
    the majority of cluster centers are pulled into the overall center of
    gravity. If you are clustering data with very high dimensionality and
    encounter this issue, another clustering method may be required. For more
    information and the theory behind this, see Winkler et al. [2]_.
    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.
    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high
           dimensional spaces. 2012. Contemporary Theory and Pragmatic
           Approaches in Fuzzy Computing Utilization, 1.
    """
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjm, d] = _cmeans0(data, u2, c, m)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u  


plt.figure(figsize=(12, 12))

n_clusters = 4
m = 2
X = mu1
centers, L = cmeans(X.T, n_clusters, m)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=L[0,:])
plt.title("No Error")
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=L[1,:])
plt.title("Error 1")
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=L[2,:])
plt.title("Error 2")
plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=L[3,:])
plt.title("Error 3")
plt.savefig(fcmname)


