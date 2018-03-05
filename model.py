import sys, time, argparse, os, re
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import batch_norm
import tensorflow.contrib.slim as slim
import tqdm
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

_VALIDATION_RATIO = 0.1

class MEDGAN(object):
    def __init__(self,
                 sess,
                 model_name='medGAN',
                 dataType='binary',
                 inputDim=615,
                 embeddingDim=128,
                 randomDim=128,
                 generatorDims=(128, 128),
                 discriminatorDims=(256, 128, 1),
                 compressDims=(),
                 decompressDims=(),
                 bnDecay=0.99,
                 l2scale=0.001):
        self.sess = sess
        self.model_name=model_name
        ## create a dedicated folder for this model
        if os.path.exists(self.model_name):
            print('WARNING: the folder "{}" already exists!'.format(self.model_name))
        else:
            os.makedirs(self.model_name)
            os.makedirs(os.path.join(self.model_name, 'outputs'))
            os.makedirs(os.path.join(self.model_name, 'models'))
        self.inputDim = inputDim
        self.embeddingDim = embeddingDim
        self.generatorDims = list(generatorDims) + [embeddingDim]
        self.randomDim = randomDim
        self.dataType = dataType

        if dataType == 'binary':
            self.aeActivation = tf.nn.tanh
        else:
            self.aeActivation = tf.nn.relu

        self.generatorActivation = tf.nn.relu
        self.discriminatorActivation = tf.nn.relu
        self.discriminatorDims = discriminatorDims
        self.compressDims = list(compressDims) + [embeddingDim]
        self.decompressDims = list(decompressDims) + [inputDim]
        self.bnDecay = bnDecay
        self.l2scale = l2scale
        
    def build_model(self):
        ## input variables
        self.x_raw = tf.placeholder('float', [None, self.inputDim])
        self.x_random= tf.placeholder('float', [None, self.randomDim])
        self.keep_prob = tf.placeholder('float')
        self.bn_train = tf.placeholder('bool')
        ## loss functions
        self.loss_ae, self.decodeVariables = self.buildAutoencoder(self.x_raw)
        self.x_fake = self.buildGenerator(self.x_random, self.bn_train)
        self.loss_d, self.loss_g, self.y_hat_real, self.y_hat_fake = \
            self.buildDiscriminator(self.x_raw, self.x_fake, self.keep_prob, self.decodeVariables, self.bn_train)
        ## trainable variables
        t_vars = tf.trainable_variables()
        self.ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        ## model saver
        self.saver = tf.train.Saver(max_to_keep = None) ## keep all checkpoints!
    
    def loadData(self, dataPath=''):
        data = np.load(dataPath)

        if self.dataType == 'binary':
            data = np.clip(data, 0, 1)

        trainX, validX = train_test_split(data, test_size=_VALIDATION_RATIO)
        return trainX, validX
    
    def buildAutoencoder(self, x_input):
        decodeVariables = {}
        with tf.variable_scope('autoencoder', regularizer=l2_regularizer(self.l2scale)):
            tempVec = x_input
            tempDim = self.inputDim
            i = 0
            for compressDim in self.compressDims:
                W = tf.get_variable('aee_W_'+str(i), shape=[tempDim, compressDim])
                b = tf.get_variable('aee_b_'+str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = compressDim
                i += 1
    
            i = 0
            for decompressDim in self.decompressDims[:-1]:
                W = tf.get_variable('aed_W_'+str(i), shape=[tempDim, decompressDim])
                b = tf.get_variable('aed_b_'+str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = decompressDim
                decodeVariables['aed_W_'+str(i)] = W
                decodeVariables['aed_b_'+str(i)] = b
                i += 1
            W = tf.get_variable('aed_W_'+str(i), shape=[tempDim, self.decompressDims[-1]])
            b = tf.get_variable('aed_b_'+str(i), shape=[self.decompressDims[-1]])
            decodeVariables['aed_W_'+str(i)] = W
            decodeVariables['aed_b_'+str(i)] = b

            if self.dataType == 'binary':
                x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec,W),b))
                loss = tf.reduce_mean(-tf.reduce_sum(x_input * tf.log(x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - x_reconst + 1e-12), 1), 0)
            else:
                x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec,W),b))
                loss = tf.reduce_mean((x_input - x_reconst)**2)
            
        return loss, decodeVariables
    
    def buildGenerator(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator', regularizer=l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output
    
    def buildGeneratorTest(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator', reuse = True, regularizer=l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output
    
    def getDiscriminatorResults(self, x_input, keepRate, reuse=False):
        batchSize = tf.shape(x_input)[0]
        inputMean = tf.reshape(tf.tile(tf.reduce_mean(x_input,0), [batchSize]), (batchSize, self.inputDim))
        tempVec = tf.concat([x_input, inputMean], 1) ## Chia-Ching: change parameter order
        tempDim = self.inputDim * 2
        with tf.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, discDim])
                b = tf.get_variable('b_'+str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec,W),b))
                h = tf.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b)))
        return y_hat
    
    def buildDiscriminator(self, x_real, x_fake, keepRate, decodeVariables, bn_train):
        ## Discriminate for real samples
        y_hat_real = self.getDiscriminatorResults(x_real, keepRate, reuse=False)

        ## Decompress, then discriminate for fake samples
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        y_hat_fake = self.getDiscriminatorResults(x_decoded, keepRate, reuse=True)

        loss_d = -tf.reduce_mean(tf.log(y_hat_real + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake + 1e-12))
        loss_g = -tf.reduce_mean(tf.log(y_hat_fake + 1e-12))

        return loss_d, loss_g, y_hat_real, y_hat_fake
    
    def print2file(self, buf, logFile):
        outfd = open(logFile, 'a')
        outfd.write(buf + '\n')
        outfd.close()
    
    def generateData(self,
                     gen_from='medGAN',
                     gen_from_ckpt=None,
                     out_name='temp.npy',
                     nSamples=10000,
                     batchSize=1000):
        x_emb = self.buildGeneratorTest(self.x_random, self.bn_train)
        tempVec = x_emb
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_'+str(i)]),
                                               self.decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_'+str(i)]),
                                             self.decodeVariables['aed_b_'+str(i)]))
        else:
            x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_'+str(i)]),
                                          self.decodeVariables['aed_b_'+str(i)]))

        outputVec = []
        burn_in = 1000
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            print('burning in')
            for i in tqdm.tqdm(range(burn_in)):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                output = self.sess.run(x_reconst, feed_dict={self.x_random:randomX, self.bn_train:True})
            print('generating')
            nBatches = int(np.ceil(float(nSamples)) / float(batchSize))
            for i in tqdm.tqdm(range(nBatches)):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                output = self.sess.run(x_reconst, feed_dict={self.x_random:randomX, self.bn_train:False})
                outputVec.extend(output)

            outputMat = np.array(outputVec)
            out_path = os.path.join(gen_from, 'outputs', out_name)
            np.save(out_path, outputMat)
        else:
            print(" [*] Failed to find a checkpoint")
    
    def calculateDiscAuc(self, preds_real, preds_fake):
        preds = np.concatenate([preds_real, preds_fake], axis=0)
        labels = np.concatenate([np.ones((len(preds_real))), np.zeros((len(preds_fake)))], axis=0)
        auc = roc_auc_score(labels, preds)
        return auc
    
    def calculateDiscAccuracy(self, preds_real, preds_fake):
        total = len(preds_real) + len(preds_fake)
        hit = 0
        for pred in preds_real: 
            if pred > 0.5: hit += 1
        for pred in preds_fake: 
            if pred < 0.5: hit += 1
        acc = float(hit) / float(total)
        return acc
    
    def train(self,
              data_path='data/inpatient_train_data.npy',
              init_from=None,
              nEpochs=500,
              discriminatorTrainPeriod=2,
              generatorTrainPeriod=1,
              pretrainBatchSize=100,
              batchSize=1000,
              pretrainEpochs=100,
              saveMaxKeep=0,
              nDeleteColumns=0):
        ## regularizer
        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        ## optimizer
        optimize_ae = tf.train.AdamOptimizer().minimize(self.loss_ae + sum(all_regs), var_list=self.ae_vars)
        optimize_d = tf.train.AdamOptimizer().minimize(self.loss_d + sum(all_regs), var_list=self.d_vars)
        optimize_g = tf.train.AdamOptimizer().minimize(self.loss_g + sum(all_regs), \
                                                       var_list = self.g_vars + list(self.decodeVariables.values()))
        ## load data
        trainX, validX = self.loadData(data_path)
        trainX = np.delete(trainX, range(nDeleteColumns), 1)
        validX = np.delete(validX, range(nDeleteColumns), 1)
        train_data_mean = np.mean(trainX, axis=0)

        initOp = tf.global_variables_initializer()
        nBatches = int(np.ceil(float(trainX.shape[0]) / float(batchSize)))
        log_path = os.path.join(self.model_name, 'models', self.model_name + '.log')

        ## initialization
        self.sess.run(initOp)
        
        ## load previous model if possible
        train_from_scratch = True
        epoch_counter = 0
        if init_from is not None:
            could_load, checkpoint_counter = self.load(init_from)
            if could_load:
                epoch_counter = checkpoint_counter
                train_from_scratch = False
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" [@] train from scratch")

        ## pre-train the autoencoder
        nTrainBatches = int(np.ceil(float(trainX.shape[0])) / float(pretrainBatchSize))
        nValidBatches = int(np.ceil(float(validX.shape[0])) / float(pretrainBatchSize))
        if train_from_scratch:
            trainLossVecList = []
            validLossVecList = []
            for epoch in range(pretrainEpochs):
                idx = np.random.permutation(trainX.shape[0])
                trainLossVec = []
                for i in tqdm.tqdm(range(nTrainBatches)):
                    batchX = trainX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                    _, loss = self.sess.run([optimize_ae, self.loss_ae], feed_dict={self.x_raw:batchX})
                    trainLossVec.append(loss)
                idx = np.random.permutation(validX.shape[0])
                validLossVec = []
                for i in tqdm.tqdm(range(nValidBatches)):
                    batchX = validX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                    loss = self.sess.run(self.loss_ae, feed_dict={self.x_raw:batchX})
                    validLossVec.append(loss)
                validReverseLoss = 0.
                buf = 'Pretrain_Epoch:%d, trainLoss:%f, validLoss:%f, validReverseLoss:%f' % \
                    (epoch, np.mean(trainLossVec), np.mean(validLossVec), validReverseLoss)
                print(buf)
                self.print2file(buf, log_path)
                trainLossVecList.append(np.mean(trainLossVec))
                validLossVecList.append(np.mean(validLossVec))
            ## plot the scatter plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(range(pretrainEpochs), trainLossVecList, 'b', alpha=0.5)
            ax.plot(range(pretrainEpochs), validLossVecList, 'r', alpha=0.5)
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.legend(loc=4)
            fig.savefig(self.model_name+'/outputs/ae_loss.png')
            plt.close(fig)

        ## the main loop of GAN
        d_loss_avg_vec = []
        g_loss_avg_vec = []
        corr_vec = []
        nzc_vec = []
        for epoch in range(nEpochs):
            ## (1) training
            idx = np.arange(trainX.shape[0])
            d_loss_vec= []
            g_loss_vec = []
            for i in tqdm.tqdm(range(nBatches)):
                for _ in range(discriminatorTrainPeriod):
                    batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                    batchX = trainX[batchIdx]
                    randomX = np.random.normal(size=(batchSize, self.randomDim))
                    _, discLoss = self.sess.run([optimize_d, self.loss_d],
                                           feed_dict={self.x_raw:batchX, self.x_random:randomX,
                                                      self.keep_prob:1.0, self.bn_train:False})
                    d_loss_vec.append(discLoss)
                for _ in range(generatorTrainPeriod):
                    randomX = np.random.normal(size=(batchSize, self.randomDim))
                    _, generatorLoss = self.sess.run([optimize_g, self.loss_g],
                                                feed_dict={self.x_raw:batchX, self.x_random:randomX,
                                                           self.keep_prob:1.0, self.bn_train:True})
                    g_loss_vec.append(generatorLoss)
            d_loss_avg_vec.append(np.mean(d_loss_vec))
            g_loss_avg_vec.append(np.mean(g_loss_vec))

            ## (2) validation
            idx = np.arange(len(validX))
            nValidBatches = int(np.ceil(float(len(validX)) / float(batchSize)))
            validAccVec = []
            validAucVec = []
            for i in tqdm.tqdm(range(nValidBatches)):
                batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                batchX = validX[batchIdx]
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                preds_real, preds_fake, = self.sess.run([self.y_hat_real, self.y_hat_fake],
                                                   feed_dict={self.x_raw:batchX, self.x_random:randomX,
                                                              self.keep_prob:1.0, self.bn_train:False})
                validAcc = self.calculateDiscAccuracy(preds_real, preds_fake)
                validAuc = self.calculateDiscAuc(preds_real, preds_fake)
                validAccVec.append(validAcc)
                validAucVec.append(validAuc)
            buf = 'Epoch:%d, d_loss:%f, g_loss:%f, accuracy:%f, AUC:%f' % \
                (epoch, np.mean(d_loss_vec), np.mean(g_loss_vec), np.mean(validAccVec), np.mean(validAucVec))
            print(buf)
            self.print2file(buf, log_path)

            ## (3) Save model weights
            ## counter for file names of saved models
            epoch_counter += 1
            if epoch % 10 == 0:
                save_path = self.saver.save(self.sess, os.path.join(self.model_name, 'models', self.model_name + '.model'),
                                            global_step=epoch_counter)
                print(save_path)
                
                ## monitor the quality of generated data during training process:
                self.generateData(nSamples=trainX.shape[0],
                                  gen_from=self.model_name,
                                  out_name='temp.npy',
                                  batchSize=1000)
                temp_data = np.load(self.model_name+'/outputs/temp.npy')
                temp_data_mean = np.mean(temp_data, axis=0)
                ## compute the correlation and number of all-zero columns
                corr = pearsonr(temp_data_mean, train_data_mean)
                corr_vec.append(corr[0])
                temp_data_sum = np.sum(temp_data, axis=0)
                nzc = np.sum(temp_data_sum[i] > 0 for i in range(temp_data_sum.shape[0]))
                nzc_vec.append(nzc)
                print('corr = {}, none-zero columns: {}'.format(corr, nzc))
                ## plot the scatter plot
                fig, ax = plt.subplots(figsize=(8,6))
                slope, intercept = np.polyfit(train_data_mean, temp_data_mean, 1)
                fitted_values = [slope * i + intercept for i in train_data_mean]
                identity_values = [1 * i + 0 for i in train_data_mean]
                ax.plot(train_data_mean, fitted_values, 'b', alpha=0.5)
                ax.plot(train_data_mean, identity_values, 'r', alpha=0.5)
                ax.scatter(train_data_mean, temp_data_mean, alpha=0.3)
                ax.set_title('Epoch: %d, corr: %.4f, none-zero columns: %d'%(epoch, corr[0], nzc))
                ax.set_xlabel('real')
                ax.set_ylabel('generated')
                fig.savefig(self.model_name+'/outputs/{}.png'.format(epoch))
                plt.close(fig)

        return [d_loss_avg_vec, g_loss_avg_vec, corr_vec, nzc_vec]
    
    def load(self, init_from, init_from_ckpt = None):
        ckpt = tf.train.get_checkpoint_state(os.path.join(init_from, 'models'))
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver.restore(self.sess, os.path.join(init_from, 'models', ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

class MEDWGAN(MEDGAN):
    ## Reuse init function of MEDGAN
    def __init__(self,
                 sess,
                 model_name='medGAN',
                 dataType='binary',
                 inputDim=615,
                 embeddingDim=128,
                 randomDim=128,
                 generatorDims=(128, 128),
                 discriminatorDims=(256, 128, 1),
                 compressDims=(),
                 decompressDims=(),
                 bnDecay=0.99,
                 l2scale=0.001,
                 gp_scale=10.0):
        super(MEDWGAN, self).__init__(sess,
                                      model_name,
                                      dataType,
                                      inputDim,
                                      embeddingDim,
                                      randomDim,
                                      generatorDims,
                                      discriminatorDims,
                                      compressDims,
                                      decompressDims,
                                      bnDecay,
                                      l2scale)
        self.gp_scale = gp_scale
    
    ## Re-write getDiscriminatorResults(), buildDiscriminator(), and train() functions
    def getDiscriminatorResults(self, x_input, keepRate, reuse=False):
        batchSize = tf.shape(x_input)[0]
        inputMean = tf.reshape(tf.tile(tf.reduce_mean(x_input,0), [batchSize]), (batchSize, self.inputDim))
        tempVec = tf.concat([x_input, inputMean], 1) ## Chia-Ching: change parameter order
        tempDim = self.inputDim * 2
        with tf.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, discDim])
                b = tf.get_variable('b_'+str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec,W),b))
                h = tf.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.add(tf.matmul(tempVec, W), b)) ## WGAN: remove tf.nn.sigmoid()
        return y_hat
    
    def buildDiscriminator(self, x_real, x_fake, keepRate, decodeVariables, bn_train):
        ## Discriminate for real samples
        y_hat_real = self.getDiscriminatorResults(x_real, keepRate, reuse=False)

        ## Decompress, then discriminate for fake samples
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        y_hat_fake = self.getDiscriminatorResults(x_decoded, keepRate, reuse=True)
        
        ## WGAN: remove tf.log()
        loss_d = tf.reduce_mean(y_hat_fake) - tf.reduce_mean(y_hat_real)
        loss_g = -tf.reduce_mean(y_hat_fake)
        
        ## improved W-GAN (gradient penalty)
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * x_real + (1 - epsilon) * x_decoded
        d_hat = self.getDiscriminatorResults(x_hat, keepRate, reuse=True)
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.gp_scale)
        loss_d = loss_d + ddx

        return loss_d, loss_g, y_hat_real, y_hat_fake
    
    def train(self,
              data_path='data/inpatient_train_data.npy',
              init_from=None,
              nEpochs=500,
              discriminatorTrainPeriod=2,
              generatorTrainPeriod=1,
              pretrainBatchSize=100,
              batchSize=1000,
              pretrainEpochs=100,
              saveMaxKeep=0,
              nDeleteColumns=0):
        ## regularizer
        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        ## optimizer
        optimize_ae = tf.train.AdamOptimizer().minimize(self.loss_ae + sum(all_regs), var_list=self.ae_vars)
        optimize_d = tf.train.AdamOptimizer().minimize(self.loss_d + sum(all_regs), var_list=self.d_vars)
        ## Chia-Ching: list(decodeVariables.values())
        optimize_g = tf.train.AdamOptimizer().minimize(self.loss_g + sum(all_regs), \
                                                       var_list = self.g_vars + list(self.decodeVariables.values()))
        ## load data
        trainX, validX = self.loadData(data_path)
        trainX = np.delete(trainX, range(nDeleteColumns), 1)
        validX = np.delete(validX, range(nDeleteColumns), 1)
        train_data_mean = np.mean(trainX, axis=0)

        initOp = tf.global_variables_initializer()
        nBatches = int(np.ceil(float(trainX.shape[0]) / float(batchSize)))
        log_path = os.path.join(self.model_name, 'models', self.model_name + '.log')
        
        ## initialization
        self.sess.run(initOp)
        
        ## load previous model if possible
        train_from_scratch = True
        epoch_counter = 0
        if init_from is not None:
            could_load, checkpoint_counter = self.load(init_from)
            if could_load:
                epoch_counter = checkpoint_counter
                train_from_scratch = False
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" [@] train from scratch")

        ## pre-train the autoencoder
        nTrainBatches = int(np.ceil(float(trainX.shape[0])) / float(pretrainBatchSize))
        nValidBatches = int(np.ceil(float(validX.shape[0])) / float(pretrainBatchSize))
        if train_from_scratch:
            trainLossVecList = []
            validLossVecList = []
            for epoch in range(pretrainEpochs):
                idx = np.random.permutation(trainX.shape[0])
                trainLossVec = []
                for i in tqdm.tqdm(range(nTrainBatches)):
                    batchX = trainX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                    _, loss = self.sess.run([optimize_ae, self.loss_ae], feed_dict={self.x_raw:batchX})
                    trainLossVec.append(loss)
                idx = np.random.permutation(validX.shape[0])
                validLossVec = []
                for i in tqdm.tqdm(range(nValidBatches)):
                    batchX = validX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                    loss = self.sess.run(self.loss_ae, feed_dict={self.x_raw:batchX})
                    validLossVec.append(loss)
                validReverseLoss = 0.
                buf = 'Pretrain_Epoch:%d, trainLoss:%f, validLoss:%f, validReverseLoss:%f' % \
                    (epoch, np.mean(trainLossVec), np.mean(validLossVec), validReverseLoss)
                print(buf)
                self.print2file(buf, log_path)
                trainLossVecList.append(np.mean(trainLossVec))
                validLossVecList.append(np.mean(validLossVec))
            ## plot the scatter plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(range(pretrainEpochs), trainLossVecList, 'b', alpha=0.5)
            ax.plot(range(pretrainEpochs), validLossVecList, 'r', alpha=0.5)
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.legend(loc=4)
            fig.savefig(self.model_name+'/outputs/ae_loss.png')
            plt.close(fig)

        ## the main loop of GAN
        d_loss_avg_vec = []
        g_loss_avg_vec = []
        corr_vec = []
        nzc_vec = []
        for epoch in range(nEpochs):
            ## (1) training
            idx = np.arange(trainX.shape[0])
            d_loss_vec= []
            g_loss_vec = []
            for i in tqdm.tqdm(range(nBatches)):
                for _ in range(discriminatorTrainPeriod):
                    batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                    batchX = trainX[batchIdx]
                    randomX = np.random.normal(size=(batchSize, self.randomDim))
                    _, discLoss = self.sess.run([optimize_d, self.loss_d],
                                           feed_dict={self.x_raw:batchX, self.x_random:randomX,
                                                      self.keep_prob:1.0, self.bn_train:False})
                    d_loss_vec.append(discLoss)
                for _ in range(generatorTrainPeriod):
                    randomX = np.random.normal(size=(batchSize, self.randomDim))
                    _, generatorLoss = self.sess.run([optimize_g, self.loss_g],
                                                feed_dict={self.x_raw:batchX, self.x_random:randomX,
                                                           self.keep_prob:1.0, self.bn_train:True})
                    g_loss_vec.append(generatorLoss)
            d_loss_avg_vec.append(np.mean(d_loss_vec))
            g_loss_avg_vec.append(np.mean(g_loss_vec))

            ## (2) validation
            idx = np.arange(len(validX))
            nValidBatches = int(np.ceil(float(len(validX)) / float(batchSize)))
            validAccVec = []
            validAucVec = []
            for i in tqdm.tqdm(range(nValidBatches)):
                batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                batchX = validX[batchIdx]
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                preds_real, preds_fake, = self.sess.run([self.y_hat_real, self.y_hat_fake],
                                                   feed_dict={self.x_raw:batchX, self.x_random:randomX,
                                                              self.keep_prob:1.0, self.bn_train:False})
                ## Outputs of discriminator in WGAN are real numbers instead of probabilities.
                ## ==> Add sigmoid transformation here for validation.
                preds_real = 1 / (1 + np.exp(-preds_real))
                preds_fake = 1 / (1 + np.exp(-preds_fake))
                validAcc = self.calculateDiscAccuracy(preds_real, preds_fake)
                validAuc = self.calculateDiscAuc(preds_real, preds_fake)
                validAccVec.append(validAcc)
                validAucVec.append(validAuc)
            buf = 'Epoch:%d, d_loss:%f, g_loss:%f, accuracy:%f, AUC:%f' % \
                (epoch, np.mean(d_loss_vec), np.mean(g_loss_vec), np.mean(validAccVec), np.mean(validAucVec))
            print(buf)
            self.print2file(buf, log_path)

            ## (3) Save model weights
            ## counter for file names of saved models
            epoch_counter += 1
            if epoch % 10 == 0:
                save_path = self.saver.save(self.sess, os.path.join(self.model_name, 'models', self.model_name + '.model'),
                                            global_step=epoch_counter)
                print(save_path)
                
                ## monitor the quality of generated data during training process:
                self.generateData(nSamples=trainX.shape[0],
                                  gen_from=self.model_name,
                                  out_name='temp.npy',
                                  batchSize=1000)
                temp_data = np.load(self.model_name+'/outputs/temp.npy')
                temp_data_mean = np.mean(temp_data, axis=0)
                ## compute the correlation and number of all-zero columns
                corr = pearsonr(temp_data_mean, train_data_mean)
                corr_vec.append(corr[0])
                temp_data_sum = np.sum(temp_data, axis=0)
                nzc = np.sum(temp_data_sum[i] > 0 for i in range(temp_data_sum.shape[0]))
                nzc_vec.append(nzc)
                print('corr = {}, none-zero columns: {}'.format(corr, nzc))
                ## plot the scatter plot
                fig, ax = plt.subplots(figsize=(8,6))
                slope, intercept = np.polyfit(train_data_mean, temp_data_mean, 1)
                fitted_values = [slope * i + intercept for i in train_data_mean]
                identity_values = [1 * i + 0 for i in train_data_mean]
                ax.plot(train_data_mean, fitted_values, 'b', alpha=0.5)
                ax.plot(train_data_mean, identity_values, 'r', alpha=0.5)
                ax.scatter(train_data_mean, temp_data_mean, alpha=0.3)
                ax.set_title('Epoch: %d, corr: %.4f, none-zero columns: %d'%(epoch, corr[0], nzc))
                ax.set_xlabel('real')
                ax.set_ylabel('generated')
                fig.savefig(self.model_name+'/outputs/{}.png'.format(epoch))
                plt.close(fig)
        
        return [d_loss_avg_vec, g_loss_avg_vec, corr_vec, nzc_vec]

## continuous version of boundary-seeking GAN
class MEDBGAN(MEDGAN):
    ## Reuse init function of MEDGAN
    def __init__(self,
                 sess,
                 model_name='medGAN',
                 dataType='binary',
                 inputDim=615,
                 embeddingDim=128,
                 randomDim=128,
                 generatorDims=(128, 128),
                 discriminatorDims=(256, 128, 1),
                 compressDims=(),
                 decompressDims=(),
                 bnDecay=0.99,
                 l2scale=0.001):
        super(MEDBGAN, self).__init__(sess,
                                      model_name,
                                      dataType,
                                      inputDim,
                                      embeddingDim,
                                      randomDim,
                                      generatorDims,
                                      discriminatorDims,
                                      compressDims,
                                      decompressDims,
                                      bnDecay,
                                      l2scale)
    
    def buildDiscriminator(self, x_real, x_fake, keepRate, decodeVariables, bn_train):
        ## Discriminate for real samples
        y_hat_real = self.getDiscriminatorResults(x_real, keepRate, reuse=False)

        ## Decompress, then discriminate for fake samples
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        y_hat_fake = self.getDiscriminatorResults(x_decoded, keepRate, reuse=True)

        loss_d = -tf.reduce_mean(tf.log(y_hat_real + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake + 1e-12))
        loss_g = 0.5*tf.reduce_mean(tf.square(tf.log(y_hat_fake + 1e-12) - tf.log(1. - y_hat_fake + 1e-12)))

        return loss_d, loss_g, y_hat_real, y_hat_fake
