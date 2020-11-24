#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:11:10 2019

@author: ubuntu
"""

import tensorflow.contrib.layers as tcl
from ops import *
import tensorflow as tf 

class LatentDiscriminator(object):
    def __init__(self, name='LatentDiscriminator', y_dim = 530):
        self.name = name
        self.y_dim = y_dim

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = tcl.fully_connected(inputs, 256, activation_fn=tf.nn.relu)
            out = tcl.fully_connected(out, self.y_dim, activation_fn=None)
            return out
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class encoder_style_SPADE(object):
    #dis_as_v = []
    def __init__(self, in_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'styleganencoder'
        self.in_dim = in_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi 
        
    def enc_adain_oir(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'spade') as scope:
            input_shape = x.shape.as_list()
            mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
            x_ = (x - mu_x) / (sigma_x + 1e-6)

            conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
            mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
            sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')

            out = sigma_y * x_ + mu_y
            return out
        
    def do_mynorm(self, inputs, atts, name=None, is_training=False):  # only z, no label
        input_shape = inputs.shape.as_list()
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs - mu_x) / (sigma_x + 1e-6)
        mu_y = atts[0]
        sigma_y = atts[1]

        adain = sigma_y * x_ + mu_y
        return adain
    
    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label
        if len(embs_in.shape.as_list())!=2:
            embs_in = tf.reshape(embs_in, (embs_in.shape.as_list()[0], -1))
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2]*9, input_shape[3]/9))  # (N, size**2, c)
        if input_shape[3]/9!=embs_in.shape.as_list()[-1]:
            embs_in = lrelu(fully_connected(embs_in, input_shape[3]/9, use_bias=True, scope=name))  
        embs = tf.reshape(embs_in, (-1, input_shape[3]/9, 1)) 
        att = tf.matmul(input_r, embs)  # (N, size**2, 1)
        att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 9)) 
        return att_weight
    
    def enc_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'spade') as scope:
            input_shape = x.shape.as_list()  
            conv = conv_2d(x, channels=25*9, kernel=3, stride=1, pad=1, sn=False, name='conv_offset1') 
            offsetY = self.do_att(conv, latent_in[0], name='c_convY', is_training=training) 
            offsetX = self.do_att(conv, latent_in[1], name='c_convX', is_training=training) 
            offset = tf.concat([offsetY, offsetX], -1)
            out = deform_conv2d(x, offset, [3,3, input_shape[3],  input_shape[3]], activation = None, scope=None)
            out = tf.concat([x, out], -1)
            return out
 
    def __call__(self, conv, embs, batch_size, reuse=tf.AUTO_REUSE, pg=4, pa=False, t=False, is_mlp=True, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))
            if pa:
                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv4'), scope='IN_4'))

            embs = lrelu(instance_norm(
                self.enc_adain(conv, embs, training=is_training, name='enc_deformconv_0') , scope='IN_deformconv_0'))
            
            for i in range(pg):
                res = conv

                res = conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)) 
                res = self.enc_adain_oir(res, embs, training=is_training, name='spade_res%d'%i) 
                res = lrelu(res)

                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i))
                conv = self.enc_adain_oir(conv, embs, training=is_training, name='spade1%d'%i)  
                conv = lrelu(conv)

                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i))
                conv = self.enc_adain_oir(conv, embs, training=is_training, name='spade2%d'%i) 
                conv = lrelu(conv)

                conv = res + conv
            
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv_l1'), scope='IN_conv_l1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=3, stride=2, pad=1, sn=False,name='enc_conv_l2'), scope='IN_conv_l2')) 
            
            conv = tf.reshape(conv, [batch_size, -1])
#            # conv = tf.reduce_mean(conv, axis=(1, 2))
#
            out = lrelu(fully_connected(conv, 1024, use_bias=True, sn=False, scope='o_1'))
            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.in_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.in_dim, activation_fn=None)
            return z_mu, z_logvar 

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class generate_style_res(object):
    #dis_as_v = []
    def     __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage): 
        return min(512 / (2 **(stage * 1)), 256)

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list() 
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            return out
#
    def enc_adain_oir(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
            x_ = (x - mu_x) / (sigma_x + 1e-6)

            conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
            mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
            sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')

            out = sigma_y * x_ + mu_y
            return out
    
    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label
        if len(embs_in.shape.as_list())!=2:
            embs_in = tf.reshape(embs_in, (embs_in.shape.as_list()[0], -1))
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2]*9, input_shape[3]/9))  # (N, size**2, c)
        if input_shape[3]/9!=embs_in.shape.as_list()[-1]:
            embs_in = lrelu(fully_connected(embs_in, input_shape[3]/9, use_bias=True, scope=name))  
        embs = tf.reshape(embs_in, (-1, input_shape[3]/9, 1)) 
        att = tf.matmul(input_r, embs)  # (N, size**2, 1)
        att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 9)) 
        return att_weight
    
    def enc_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'spade') as scope:
            input_shape = x.shape.as_list()  
            conv = conv_2d(x, channels=25*9, kernel=3, stride=1, pad=1, sn=False, name='conv_offset1') 
            offsetY = self.do_att(conv, latent_in[0], name='c_convY', is_training=training) 
            offsetX = self.do_att(conv, latent_in[1], name='c_convX', is_training=training) 
            offset = tf.concat([offsetY, offsetX], -1)
            out = deform_conv2d(x, offset, [3,3, input_shape[3],  input_shape[3]], activation = None, scope=None)
            out = tf.concat([x, out], -1)
            return out 
    

    def __call__(self, inputs, en_embs, embs, pg=4, pa=False, t=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse == True:
            #     scope.reuse_variables()

            input_shape = inputs.shape.as_list()

            shape = inputs.shape.as_list() 
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 4, 4, -1])
            de = lrelu(conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_1_conv'))
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            de = lrelu(de)
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            de = lrelu(de)
 
            en_embs = lrelu(instance_norm(self.enc_adain(de, en_embs, training=is_training, name='gen_deformconv_0') , scope='gen_IN_deformconv_0'))

            for i in range(pg):
 
                res = de
                res = conv_2d(res, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_res_{}'.format(i))
#                res = instance_norm(res, scope='res%d'%i)
                res1 = self.layer_adain(res, embs, training=is_training, name='res%d'%i)
                res = self.enc_adain_oir(res, en_embs, training=is_training, name='spade_res%d'%i) 
                res = tf.concat((res1, res), 3)
                res = lrelu(res)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv1_{}'.format(i))
#                de = instance_norm(de, scope='style1%d'%i) 
                de1 = self.layer_adain(de, embs, training=is_training, name='style1%d'%i)
                de = self.enc_adain_oir(de, en_embs, training=is_training, name='spade_style1%d'%i) 
                de = tf.concat((de1, de), 3)
                de = lrelu(de)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv2_{}'.format(i))
#                de = instance_norm(de, scope='style2%d'%i)
                de1 = self.layer_adain(de, embs, training=is_training, name='style1%d'%i)
                de = self.enc_adain_oir(de, en_embs, training=is_training, name='spade_style2%d'%i) 
                de = tf.concat((de1, de), 3)
                de = lrelu(de)

                de = res + de

            #To RGB
            de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l1'), scope='gen_LN_l1'))
            if pa:
                de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l2'), scope='gen_LN_l2'))
            # if pa:
            #     de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(conv_2d(de, channels=self.channel * 1, kernel=7, stride=1, pad=3, sn=False, name='gen_conv_l3'), scope='gen_LN_l3'))
            de = conv_2d(de, channels=3, kernel=1, stride=1, pad=0, sn=False, name='gen_out')
            de = tanh(de)
            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


