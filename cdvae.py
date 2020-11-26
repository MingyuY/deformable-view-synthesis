import os, sys
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim 
slim = contrib_slim
import numpy as np
import matplotlib.pyplot as plt
import math
sys.path.append(os.path.dirname(__file__)) 
from models.mynets_celeb import * 
from models.cdvae import *
from models.label_embed import *
from datasets.get_data import * 
from glob import glob
from tqdm import tqdm
import argparse

def loss_hinge_dis(dis_f, dis_p, dis_fp, dis_real):
    loss = 1*tf.reduce_mean(tf.nn.relu(1. - dis_real))
    loss += tf.reduce_mean(tf.nn.relu(1. + dis_p))/3.0 
    loss += tf.reduce_mean(tf.nn.relu(1. + dis_f))/3.0 
    loss += tf.reduce_mean(tf.nn.relu(1. + dis_fp))/3.0
    return loss

def loss_hinge_gen(dis_fake):
    loss = -tf.reduce_mean(dis_fake)
    return loss

def sample_z(m, n):
    return np.random.normal(0, 1, size=[m, n])

def sample_normal(avg, log_var):
    with tf.name_scope('SampleNormal'):
        epsilon = tf.random_normal(tf.shape(avg))
        return tf.add(avg, tf.multiply(tf.exp(0.5 * log_var), epsilon))

def kl_loss(avg, log_var):
    with tf.name_scope('KLLoss'):
        return tf.reduce_mean(0.5 * tf.reduce_mean(tf.exp(log_var) + avg ** 2 - 1. - log_var, 1))


class SNGAN():
    def __init__(self, encoder, generator, discriminator, labelembXE, labelembYE, labelembXD, labelembYD, latent_discriminator, batch_size, \
                 log_dir='logs/imagenet',
                 model_dir='models/imagenet/', learn_rate_init=2e-4):
        self.log_vars = []
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.size = 128
        self.batch_size = batch_size
        self.dataset_file = ['datasets/dataset.tfrecords']
        self.img, self.label, self.p, self.t = read_and_decode_data(self.dataset_file, self.size) 
        self.img_batch, self.label_batch, self.p_batch, self.t_batch = tf.train.shuffle_batch([self.img, self.label, self.p, self.t],
                                                                                              batch_size=self.batch_size,
                                                                                              capacity=1000, min_after_dequeue=600)
        self.is_wganGP = True
        self.gp_lambda = 10
        self.learn_rate_init = learn_rate_init
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        # self.classifier = classifier
        self.labelembXE = labelembXE
        self.labelembYE = labelembYE
        self.labelembXD = labelembXD
        self.labelembYD = labelembYD
        self.latent_discriminator = latent_discriminator
        # self.data = data
        self.in_dim = 512
        self.z_dim = 256
        self.y_num = 1
        self.y_dim = 62
        self.channel = 3
        self.latent_size = 128
        self.w = 5
        self.c = 1
        self.pa = True
        self.pg = 4
        self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.size, self.size, self.channel], name='X')
        self.z_in = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_dim], name='z_in')
        self.z_p = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='z_p')
        self.z_latentXE = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_size], name='z_latentXE') #mapping
        self.z_latentXD = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_size], name='z_latentXD') #mapping
        self.z_latent1XD = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_size], name='z_latent1XD') #mapping
        self.z_latentYE = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_size], name='z_latentYE') #mapping
        self.z_latentYD = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_size], name='z_latentYD') #mapping
        self.z_latent1YD = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_size], name='z_latent1YD') #mapping
        self.Y = tf.placeholder(tf.int32, shape=[self.batch_size], name='Y')
        self.Y_rand = tf.placeholder(tf.int32, shape=[self.batch_size], name='Y_rand')
        self.training = tf.placeholder(tf.bool,  name='training')
        self.Y_onehot = tf.one_hot(self.Y, self.y_dim)
        self.Y_rand_onehot = tf.one_hot(self.Y_rand, self.y_dim)

        # embs_f = self.labelemb(self.z_latent, self.latent_size, self.batch_size, label_size=0, is_smooth=0, labels=self.Y_onehot)
        embs_fXE = self.labelembXE(self.latent_size, self.batch_size, self.z_latentXE, label_size=self.y_num*self.y_dim, labels=self.Y_onehot)
        embs_f_reshape_XE = tf.reshape(embs_fXE, [self.batch_size, self.w, self.w, self.c])
        embs_fYE = self.labelembYE(self.latent_size, self.batch_size, self.z_latentYE, label_size=self.y_num*self.y_dim, labels=self.Y_onehot)
        embs_f_reshape_YE = tf.reshape(embs_fYE, [self.batch_size, self.w, self.w, self.c])
        
        embs_fXD = self.labelembXD(self.latent_size, self.batch_size, self.z_latentXD, label_size=self.y_num*self.y_dim, labels=self.Y_onehot)
        embs_f_reshape_XD = tf.reshape(embs_fXD, [self.batch_size, self.w, self.w, self.c])
        embs_fYD = self.labelembYD(self.latent_size, self.batch_size, self.z_latentYD, label_size=self.y_num*self.y_dim, labels=self.Y_onehot)
        embs_f_reshape_YD = tf.reshape(embs_fYD, [self.batch_size, self.w, self.w, self.c])
        # embs_p = self.labelemb(self.z_latent1, self.latent_size, self.batch_size, label_size=0, is_smooth=0, labels=self.Y_rand_onehot, reuse=True)
        embs_pXD = self.labelembXD(self.latent_size, self.batch_size, self.z_latent1XD, label_size=self.y_num*self.y_dim, labels=self.Y_rand_onehot, reuse=True)
        embs_p_reshape_XD = tf.reshape(embs_pXD, [self.batch_size, self.w, self.w, self.c])
        embs_pYD = self.labelembXD(self.latent_size, self.batch_size, self.z_latent1YD,  label_size=self.y_num*self.y_dim, labels=self.Y_rand_onehot, reuse=True)
        embs_p_reshape_YD = tf.reshape(embs_pYD, [self.batch_size, self.w, self.w, self.c])

        # nets
        # self.z_mu, self.z_log_var = self.encoder(self.X, self.batch_size, labels=self.Y_onehot, is_training = self.training)
        self.z_mu, self.z_log_var = self.encoder(self.X, [embs_f_reshape_YE, embs_f_reshape_XE], self.batch_size, pg=3, pa=self.pa, labels=self.Y_onehot, is_mlp=False, is_training = self.training)
        self.z_f = sample_normal(self.z_mu, self.z_log_var)
        self.c_enc_p = self.latent_discriminator(self.z_f)

        # self.cls_real = self.classifier(self.z_f, self.batch_size)

        self.x_f = self.generator(self.z_in, [embs_f_reshape_YD, embs_f_reshape_XD], self.z_f, pa=self.pa, pg=3, is_training=self.training)
        self.x_p = self.generator(self.z_in, [embs_p_reshape_YD, embs_p_reshape_XD], self.z_f, pa=self.pa, pg=3, is_training=self.training, reuse=True)
        self.x_fp = self.generator(self.z_in, [embs_f_reshape_YD, embs_f_reshape_XD], self.z_p, pa=self.pa, pg=3, is_training=self.training, reuse=True)
        
        self.z_mu3, self.z_log_var3 = self.encoder(self.x_fp, [embs_f_reshape_YE, embs_f_reshape_XE], self.batch_size, pg=3, pa=self.pa, labels=self.Y_onehot, is_mlp=False, is_training = self.training, reuse=True)

        self.dis_real = self.discriminator(self.X, self.batch_size, labels=self.Y_onehot, pg=self.pg)
        self.dis_f = self.discriminator(self.x_f, self.batch_size, labels=self.Y_onehot, pg=self.pg, reuse = True)
        self.dis_p = self.discriminator(self.x_p, self.batch_size, labels=self.Y_rand_onehot, pg=self.pg, reuse = True)
        self.dis_fp = self.discriminator(self.x_fp, self.batch_size, labels=self.Y_onehot, pg=self.pg, reuse = True)
 
        def vgg16( inputs, reuse=False, scope='vgg_16'):
            with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],):
                                    # outputs_collections=end_points_collection):
                    self.net1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    self.net1_p = slim.max_pool2d(self.net1, [2, 2], scope='pool1')
                    self.net2 = slim.repeat(self.net1_p, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    self.net2_p = slim.max_pool2d(self.net2, [2, 2], scope='pool2')
                    self.net3 = slim.repeat(self.net2_p, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    self.net3_p = slim.max_pool2d(self.net3, [2, 2], scope='pool3')
                    self.net4 = slim.repeat(self.net3_p, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    self.net4_p = slim.max_pool2d(self.net4, [2, 2], scope='pool4')
                    self.net5 = slim.repeat(self.net4_p, 3, slim.conv2d, 512, [3, 3], scope='conv5') 
                return self.net5, self.net2
        
        def gram(f):
            input_shape = f.shape.as_list()
            f = f - tf.nn.moments(f, [3], keep_dims=True)[0]
            f1 = tf.reshape(f, (input_shape[0], input_shape[1]*input_shape[2], input_shape[3]))
            cov = tf.matmul(f1, f1, transpose_a=True)
            return cov
        self.vgg_fb, self.vgg_fb2 =  vgg16(self.X)
        self.vgg_fb_f, self.vgg_fb_f2 =  vgg16(self.x_f, reuse=True)
        self.loss_rec_f = tf.reduce_mean(tf.abs(self.vgg_fb -self.vgg_fb_f)) 
        self.log_vars.append(("loss_rec_f", self.loss_rec_f))
        self.loss_rec_f2 = tf.reduce_mean(tf.abs(self.vgg_fb2 -self.vgg_fb_f2)) 
        self.log_vars.append(("loss_rec_f2", self.loss_rec_f2))
        
        
        self.loss_style_f2 = tf.reduce_mean(tf.abs(gram(self.vgg_fb2) -gram(self.vgg_fb_f2))) 
        self.log_vars.append(("loss_style_f2", self.loss_style_f2))
        
        t_vars = tf.global_variables() 
        self.vggnet_vars = [var for var in t_vars if 'vgg_16' in var.name]
        
        self.loss_kl = kl_loss(self.z_mu, self.z_log_var)
        self.log_vars.append(("loss_kl", self.loss_kl))
        self.loss_gen = (loss_hinge_gen(self.dis_f) + loss_hinge_gen(self.dis_p) + loss_hinge_gen(self.dis_fp))/3.0
        self.log_vars.append(("loss_gen", self.loss_gen))
        self.loss_dis = loss_hinge_dis(self.dis_f, self.dis_p, self.dis_fp, self.dis_real)
        self.loss_rec = tf.reduce_mean(tf.square(self.X - self.x_f))
        self.log_vars.append(("loss_rec", self.loss_rec))

        self.loss_mu_rec = tf.reduce_mean(tf.abs(self.z_p - self.z_mu3))
        self.log_vars.append(("loss_mu_rec", self.loss_mu_rec))

        if self.is_wganGP == True:
            epsilon_1 = tf.random_uniform([], 0.0, 1.0)
            interpolated = epsilon_1 * self.X + (1 - epsilon_1) * self.x_f
            self.D_logits = self.discriminator(interpolated, self.batch_size, labels=self.Y_onehot, pg=self.pg,reuse=True)

            gradients = tf.gradients(self.D_logits, interpolated, name="D_logits_intp")[0]
            grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))

            self.gp_loss_sum = self.log_vars.append(("grad_penalty", grad_penalty))
            self.loss_dis += self.gp_lambda * grad_penalty

        self.log_vars.append(("loss_dis", self.loss_dis))


         # C loss to classify c_enc_p 
        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Y_onehot, logits = self.c_enc_p))
        self.log_vars.append(("C_loss", self.C_loss))

        # adversarial loss
        self.adv_Y = tf.ones_like(self.c_enc_p, dtype=tf.float32) / tf.cast(self.y_dim, tf.float32)
        self.adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.adv_Y, logits=self.c_enc_p))
        self.log_vars.append(("adv_loss", self.adv_loss))
        
        # Optimize
        self.enc_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_init, beta1=0.0, beta2=0.9)
        self.cls_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_init, beta1=0.0, beta2=0.9)
        self.dis_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_init, beta1=0.0, beta2=0.9)
        self.gen_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_init, beta1=0.0, beta2=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.enc_solver = self.enc_opt.minimize(1*self.loss_kl + self.loss_rec*10 + 10*self.loss_rec_f + 2*self.loss_rec_f2 + \
                                                    self.adv_loss + 0.0001*self.loss_style_f2,\
                                                    var_list = self.encoder.vars+self.labelembXE.vars+self.labelembYE.vars)
            self.dis_solver = self.dis_opt.minimize(self.loss_dis, var_list = self.discriminator.vars)
            self.gen_solver = self.gen_opt.minimize(self.loss_gen + self.loss_rec*10 + 10*self.loss_rec_f + 2*self.loss_rec_f2 +\
                                                    0.0001*self.loss_style_f2 + self.loss_mu_rec,\
                                                    var_list = self.generator.vars+self.labelembXD.vars+self.labelembYD.vars)
            self.cls_solver = self.cls_opt.minimize(self.C_loss, var_list = self.latent_discriminator.vars)

        # Summary
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)
        self.summary_op = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver(max_to_keep=None)
        self.vggnet_saver = tf.train.Saver(self.vggnet_vars)

    def generation(self, sample_folder, model_path, image_path, batch_size=1):
        i = 0
        if not os.path.exists(sample_folder + '/generation'):
            os.makedirs(sample_folder+ '/generation')
        self.sess.run(tf.global_variables_initializer())
        self.vggnet_saver.restore(self.sess, 'models/vgg_16.ckpt')
        self.saver.restore(self.sess, model_path) 
        
        test_list = glob(image_path+'/*.png')
        for l in range(len(test_list)):
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess)
            img = np.array(Image.open(test_list[l]))[:,:,:-1]
            label = int(test_list[l].split('_')[2])
            X_a = np.expand_dims(img, 0)
            Y_a = np.expand_dims(label, 0)
            X_a = X_a / 255.0 * 2 - 1
            
            b, h, w, c = X_a.shape
            fig = np.zeros((h*4, w*9, 3))
            fig[:h, :w, :] = np.array(X_a[0]) 
            for k in range(31):
                Y_b = np.tile(np.expand_dims(np.array(k), 0), batch_size)
                samples = self.sess.run(self.x_p, feed_dict={self.X:X_a, self.Y:Y_a, self.Y_rand:Y_b,  
                                          self.z_latentXE: np.random.randn(self.batch_size, self.latent_size),
                                          self.z_latentYE: np.random.randn(self.batch_size, self.latent_size),
                                          self.z_latentXD: np.random.randn(self.batch_size, self.latent_size),
                                          self.z_latentYD: np.random.randn(self.batch_size, self.latent_size),
                                          self.z_latent1XD: np.random.randn(self.batch_size, self.latent_size),
                                          self.z_latent1YD: np.random.randn(self.batch_size, self.latent_size),
                                          self.z_in: sample_z(batch_size, self.in_dim),
                                          self.training:False})
                fig[h*(k/8) : h*(k/8+1), w*(k%8+1):w*(k%8+2), :] = samples[0] 
            plt.imsave('{}/generation/{}_test.png'.format(sample_folder, str(l).zfill(3) ), \
                       (255*(fig+1)/2.0).astype(np.uint8), cmap='gray') 
                 
        self.sess.close()

    def train(self, sample_folder, training_iters=50000, batch_size=64, n_dis = 1, n_samples = 12, restore=False):
        i = 0
        self.sess.run(tf.global_variables_initializer())
        self.vggnet_saver.restore(self.sess, 'models/vgg_16.ckpt')
        restore_iter = 0
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.model_dir)
                print 'Restoring from {}...'.format(ckpt.model_checkpoint_path),
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[1]
                restore_iter = int(stem.split('-')[-1])
                i = restore_iter / 500
                print 'done'
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess)
        for iter in range(restore_iter, training_iters): 
            X_b, Y_b = self.sess.run([self.img_batch, self.label_batch])
            X_b = X_b / 255.0 * 2 - 1 
            order = np.arange(batch_size)
            np.random.shuffle(order)
            Y_b_shuf = Y_b[order]
            for _ in range(n_dis):
                kl, mu, sigma, _ = self.sess.run([self.loss_kl, self.z_mu, self.z_log_var, self.dis_solver], 
                                                 feed_dict={self.X:X_b, self.Y:Y_b, self.z_p:sample_z(batch_size, self.z_dim),
                                                          self.z_latentXE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentXD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1XD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1YD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_in: sample_z(batch_size, self.in_dim),
                                                          self.Y_rand:Y_b_shuf, self.training:True})
                self.sess.run(self.cls_solver, feed_dict={self.X:X_b, self.Y:Y_b, 
                                                          self.z_p: sample_z(batch_size, self.z_dim),
                                                          self.z_latentXE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentXD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1XD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1YD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_in: sample_z(batch_size, self.in_dim),
                                                          self.Y_rand: Y_b_shuf, self.training:True})
            for _ in range(1):
                self.sess.run(self.gen_solver,feed_dict={self.X:X_b, self.Y:Y_b, self.z_p: sample_z(batch_size, self.z_dim),
                                                          self.z_in: sample_z(batch_size, self.in_dim), 
                                                          self.z_latentXE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentXD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1XD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1YD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.Y_rand: Y_b_shuf, self.training:True})
                                                     
                self.sess.run(self.enc_solver,feed_dict={self.X:X_b, self.Y:Y_b, self.z_p: sample_z(batch_size, self.z_dim), 
                                                          self.z_latentXE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentXD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1XD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1YD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_in: sample_z(batch_size, self.in_dim),
                                                          self.Y_rand: Y_b_shuf, self.training:True})
            # summary
            summary_str = self.sess.run(self.summary_op,feed_dict={self.X: X_b, self.Y: Y_b, self.z_p: sample_z(batch_size, self.z_dim), 
                                                          self.z_latentXE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentXD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1XD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1YD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_in: sample_z(batch_size, self.in_dim),
                                                          self.Y_rand: Y_b_shuf, self.training:False})
            self.summary_writer.add_summary(summary_str, iter)

             #print loss
            if iter % 100 == 0 or iter < 100:
                fetch_list = [self.loss_kl, self.loss_rec_f, self.loss_rec_f2, self.loss_style_f2,self.loss_rec, self.loss_dis, self.loss_gen,\
                              self.z_mu, self.z_log_var, self.loss_mu_rec ]
                loss_kl_curr, loss_rec_f_curr, loss_rec_f2_curr, loss_style_f2_curr, loss_rec_curr, loss_dis_curr, loss_gen_curr,\
                z_mu, z_log_var, loss_mu_rec_curr  = self.sess.run(fetch_list,
                                    feed_dict={self.X: X_b, self.Y: Y_b, self.z_p: sample_z(batch_size, self.z_dim), 
                                                          self.z_latentXE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYE: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentXD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latentYD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1XD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_latent1YD: np.random.randn(self.batch_size, self.latent_size),
                                                          self.z_in: sample_z(batch_size, self.in_dim),
                                                          self.Y_rand: Y_b_shuf, self.training:False}) 


                print('Iter: {}; loss_kl: {:.4};  loss_rec: {:.4}; loss_rec_f: {:.4}; loss_rec_f2: {:.4}; loss_style_f2: {:.4};  loss_dis: {:.4}; \
                      loss_gen: {:.4} ; loss_mu_rec: {:.4} ;'.
                      format(iter, loss_kl_curr, loss_rec_curr, loss_rec_f_curr, loss_rec_f2_curr, loss_style_f2_curr, loss_dis_curr, loss_gen_curr,\
                             loss_mu_rec_curr))
                if iter % 500 == 0:
                    samples = self.sess.run(self.x_p, feed_dict={self.X:X_b, 
                                                                 self.Y:Y_b, 
                                                                 self.z_latentXE: np.random.randn(self.batch_size, self.latent_size),
                                                                 self.z_latentYE: np.random.randn(self.batch_size, self.latent_size),
                                                                 self.z_latentXD: np.random.randn(self.batch_size, self.latent_size),
                                                                 self.z_latentYD: np.random.randn(self.batch_size, self.latent_size),
                                                                 self.z_latent1XD: np.random.randn(self.batch_size, self.latent_size),
                                                                 self.z_latent1YD: np.random.randn(self.batch_size, self.latent_size),
                                                                 self.z_in: sample_z(batch_size, self.in_dim), 
                                                                 self.Y_rand: Y_b_shuf, self.training:False})
                    fig = data2fig(samples[:n_samples], self.size)
                    plt.savefig('{}/{}.png'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)
                    
                if (iter % 1000 == 0) or iter == training_iters - 1:
                    
                    save_path = self.model_dir + "model.ckpt"
                    self.saver.save(self.sess, save_path, global_step=iter)
        self.sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='datasets/dataset.tfrecords', help='datasets path')
    parser.add_argument('--y_dim', type=int, default=62, help='label dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--sample_folder', type=str, default='Samples/CDVAE', help='save images')
    parser.add_argument('--log_dir', type=str, default='logs/CDVAE', help='save logs')
    parser.add_argument('--model_dir', type=str, default='models/CDVAE/', help='save models')
    parser.add_argument('--mode', default='training', choices=['training', 'generation'])
    parser.add_argument('--training_iters', type=int, default=100000, help='MAX training iters')
    parser.add_argument('--model_path', type=str, default=None, help='reload model path')
    parser.add_argument('--image_path', type=str, default='datasets/generation', help='generation image path')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sample_folder = args.sample_folder
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
 
    y_dim = args.y_dim 
    encoder = encoder_style_SPADE(y_num = 1, y_dim=y_dim, channel=32)
    discriminator = stylegandiscriminate_noc(y_num = 1, y_dim=y_dim) 
    generator = generate_style_res(y_num = 1, y_dim=y_dim, channel=32)
    labelembXE = Mapping(channel=25, f_num=1, category = y_dim)
    labelembYE = Mapping(channel=25, f_num=1, category = y_dim)
    labelembXD = Mapping(channel=25, f_num=1, category = y_dim)
    labelembYD = Mapping(channel=25, f_num=1, category = y_dim)
    latent_discriminator = LatentDiscriminator(y_dim = y_dim)
    batch_size = args.batch_size
    gan = SNGAN(encoder, generator, discriminator, labelembXE, labelembYE, labelembXD, labelembYD, latent_discriminator,\
                batch_size, log_dir=args.log_dir, model_dir=args.model_dir)
    if args.mode=='training':
        gan.train(sample_folder, batch_size=batch_size, training_iters=args.training_iters, restore = False)
    elif args.mode=='generation': 
        gan.generation(sample_folder, model_path=args.model_path, image_path=args.image_path, batch_size=batch_size)
        
        
        
