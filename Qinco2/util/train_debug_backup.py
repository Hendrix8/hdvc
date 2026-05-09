# coding = utf-8

import imp
import os
import gc
import logging  
from pathlib import Path
from timeit import default_timer as timer

import torch
import numpy as np
# from ray import tune
from torch import nn, optim
from torch.utils.data import DataLoader
from lib.mpmath.functions.functions import re

from model.build import getModel
from model.initialize import LSUVinit
from model.loss import BCELoss, FbetaLoss
from util.conf import Conf
from util.data import Samples, SamplesLabelsWeights, normalize
from util.evaluate import Fbeta
from util.commons import discretize
from util.sample import Sampler
from lib.mpmath import mp
from util import autograd_hacks  


# class EEGTrainable(tune.Trainable):
class EEGTrainable():
    def __init__(self, config = None, train: bool = True):
        if config is not None:
            if type(config) is Conf:
                self.__conf = config
            elif type(config) is dict:
                # for ray tune
                self.__conf = config['not2tune']

                assert type(self.__conf) is Conf

                self.__conf.updateConf(config)
            else:
                raise ValueError('invalid config type: {:s}'.format(type(config)))
        else:
            self.__conf = Conf()

        self.__device = self.__conf.getHP('device')
        self.__batch_size = self.__conf.getHP('batch_size')

        self.__threshold = self.__conf.getHP('threshold')
        
        self.model = getModel(self.__conf)
        self.model.to(self.__device)

        if train:
            self.__conf.setup()
            self.__setup()
        else:
            checkpoint_filepath = self.__conf.getHP('checkpoint_filepath')
            assert os.path.isfile(checkpoint_filepath)
            self.__load(checkpoint_filepath)

        gc.collect()
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def __load(self, checkpoint_filepath: str):
        self.model.load_state_dict(torch.load(checkpoint_filepath))


    def __setup(self):
        logging.basicConfig(level=logging.DEBUG)
        self.__logger = logging.getLogger(self.__class__.__name__)

        filehandler = logging.FileHandler(self.__conf.getHP('log_filepath'), 'a+')
        formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname).3s [%(filename)s:%(lineno)d] %(message)s', datefmt='%m/%d/%Y:%I:%M:%S')
        filehandler.setFormatter(formatter)
        # filehandler.setLevel(logging.DEBUG)

        for handler in self.__logger.handlers[:]:
            self.__logger.removeHandler(handler)
        self.__logger.addHandler(filehandler)

        self.__max_epoch = self.__conf.getHP('num_epoch')
        self.iteration = 0

        if type(self.__conf.getHP('train_positive_samples')) is str:
            assert os.path.isfile(self.__conf.getHP('train_positive_samples'))
            train_positive_samples = np.fromfile(self.__conf.getHP('train_positive_samples'), dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')])
        elif type(self.__conf.getHP('train_positive_samples')) is list:
            train_positive_samples = []

            for tp_sample_path in self.__conf.getHP('train_positive_samples'):
                assert os.path.isfile(tp_sample_path)
                train_positive_samples.append(np.fromfile(tp_sample_path, dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')]))

            train_positive_samples = np.concatenate(train_positive_samples)
        else:
            assert type(self.__conf.getHP('train_positive_samples')) is np.ndarray
            train_positive_samples = self.__conf.getHP('train_positive_samples')

        if type(self.__conf.getHP('train_negative_samples')) is str:
            assert os.path.isfile(self.__conf.getHP('train_negative_samples'))
            train_negative_samples = np.fromfile(self.__conf.getHP('train_negative_samples'), dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')])
        elif type(self.__conf.getHP('train_negative_samples')) is list:
            train_negative_samples = []

            for tn_sample_path in self.__conf.getHP('train_negative_samples'):
                assert os.path.isfile(tn_sample_path)
                train_negative_samples.append(np.fromfile(tn_sample_path, dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')]))

            train_negative_samples = np.concatenate(train_negative_samples)
        else:
            assert type(self.__conf.getHP('train_negative_samples')) is np.ndarray
            train_negative_samples = self.__conf.getHP('train_negative_samples')

        if type(self.__conf.getHP('validate_positive_samples')) is str:
            assert os.path.isfile(self.__conf.getHP('validate_positive_samples'))
            validate_positive_samples = np.fromfile(self.__conf.getHP('validate_positive_samples'), dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')])
        elif type(self.__conf.getHP('validate_positive_samples')) is list:
            validate_positive_samples = []

            for vp_sample_path in self.__conf.getHP('validate_positive_samples'):
                assert os.path.isfile(vp_sample_path)
                validate_positive_samples.append(np.fromfile(vp_sample_path, dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')]))

            validate_positive_samples = np.concatenate(validate_positive_samples)
        else:
            assert type(self.__conf.getHP('validate_positive_samples')) is np.ndarray
            validate_positive_samples = self.__conf.getHP('validate_positive_samples')

        if type(self.__conf.getHP('validate_negative_samples')) is str:
            assert os.path.isfile(self.__conf.getHP('validate_negative_samples'))
            validate_negative_samples = np.fromfile(self.__conf.getHP('validate_negative_samples'), dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')])
        elif type(self.__conf.getHP('validate_negative_samples')) is list:
            validate_negative_samples = []

            for vn_sample_path in self.__conf.getHP('validate_negative_samples'):
                assert os.path.isfile(vn_sample_path)
                validate_negative_samples.append(np.fromfile(vn_sample_path, dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')]))

            validate_negative_samples = np.concatenate(validate_negative_samples)
        else:
            assert type(self.__conf.getHP('validate_negative_samples')) is np.ndarray
            validate_negative_samples = self.__conf.getHP('validate_negative_samples')

        assert train_positive_samples is not None and train_negative_samples is not None
        assert validate_positive_samples is not None and validate_negative_samples is not None

        train_positive_samples = np.asarray(train_positive_samples)
        train_negative_samples = np.asarray(train_negative_samples)
        validate_positive_samples = np.asarray(validate_positive_samples)
        validate_negative_samples = np.asarray(validate_negative_samples)

        assert len(train_positive_samples.shape) == len(train_negative_samples.shape) == 3
        assert len(validate_positive_samples.shape) == len(validate_negative_samples.shape) == 3

        assert 0 < train_positive_samples.shape[0] <= train_negative_samples.shape[0]
        assert 0 < validate_positive_samples.shape[0] <= validate_negative_samples.shape[0]

        assert train_positive_samples.shape[1] == train_negative_samples.shape[1] == self.__conf.getHP('num_input_channels')
        assert validate_positive_samples.shape[1] == validate_negative_samples.shape[1] == self.__conf.getHP('num_input_channels')

        assert train_positive_samples.shape[2] == train_negative_samples.shape[2] == self.__conf.getHP('dim_series')
        assert validate_positive_samples.shape[2] == validate_negative_samples.shape[2] == self.__conf.getHP('dim_series')

        assert self.__conf.getHP('num_class') == 2

        if self.__conf.getHP('num_samples_method') == 'balanced':
            self.__num_train_negative_samples = train_positive_samples.shape[0]
        elif self.__conf.getHP('num_samples_method') == 'cvpr19':
            N = mp.mpf(train_positive_samples.shape[0] + train_negative_samples.shape[0])
            beta = (N - mp.mpf(1.0)) / N
            effective = (mp.mpf(1.0) - beta ** mp.mpf(train_negative_samples.shape[0])) / (mp.mpf(1.0) - beta)
            self.__num_train_negative_samples = int(np.rint(np.float64(effective)))
        else:
            self.__num_train_negative_samples = train_negative_samples.shape[0]
        
        self.__conf.setHP('num_training_negative_samples', self.__num_train_negative_samples)
        assert train_positive_samples.shape[0] <= self.__num_train_negative_samples <= train_negative_samples.shape[0]

        if 'f1' in self.__conf.getHP('loss_function'):
            assert self.__conf.getHP('f_beta') > 0

            if self.__conf.getHP('loss_function') == 'sasu':
                self.__conf.setHP('f_alpha', train_negative_samples.shape[0] / self.__num_train_negative_samples)

        if self.__conf.getHP('normalize') == 'data':
            mu = self.__conf.getHP('mu')
            sigma = self.__conf.getHP('sigma')

            train_positive_samples = normalize(train_positive_samples, mu=mu, sigma=sigma)
            train_negative_samples = normalize(train_negative_samples, mu=mu, sigma=sigma)
            validate_positive_samples = normalize(validate_positive_samples, mu=mu, sigma=sigma)
            validate_negative_samples = normalize(validate_negative_samples, mu=mu, sigma=sigma)

        self.__num_train_positive_samples = train_positive_samples.shape[0]
        self.__num_train_negative_samples = self.__conf.getHP('num_training_negative_samples')
        self.__total_num_train_negative_samples = train_negative_samples.shape[0]
        num_train_samples = self.__num_train_positive_samples + self.__num_train_negative_samples

        self.__train_positive_samples = torch.from_numpy(np.asarray(train_positive_samples)).to(self.__device)
        self.__train_negative_samples = torch.from_numpy(np.asarray(train_negative_samples)).to(self.__device)
        self.__all_train_samples = torch.cat([self.__train_positive_samples, self.__train_negative_samples])

        self.__train_labels = torch.zeros(num_train_samples, device=self.__device, requires_grad=False)
        self.__train_labels[: self.__num_train_positive_samples] = 1

        self.__train_weights = None
    
        if self.__conf.getHP('loss_function') == 'wce':
            self.__train_weights = torch.ones(num_train_samples, device=self.__device, requires_grad=False)

            # if self.__conf.getHP('num_samples_method') == 'cvpr19': # TODO retrain
            pweight = 1 / self.__num_train_positive_samples
            nweight = 1 / self.__num_train_negative_samples

            # normalize to make 1 mean
            self.__train_weights[: self.__num_train_positive_samples] = 2 * pweight / (pweight + nweight)
            self.__train_weights[self.__num_train_positive_samples: ] = 2 * nweight / (pweight + nweight)

        if self.__num_train_negative_samples == self.__total_num_train_negative_samples:
            self.__train_dataloader = DataLoader(SamplesLabelsWeights(self.__all_train_samples, 
                                                                      self.__train_labels, 
                                                                      self.__train_weights), 
                                                 batch_size=self.__batch_size, shuffle=True)

        validate_positive_samples = torch.from_numpy(np.asarray(validate_positive_samples)).to(self.__device)
        validate_negative_samples = torch.from_numpy(np.asarray(validate_negative_samples)).to(self.__device)

        validate_samples = torch.cat([validate_positive_samples, validate_negative_samples])

        self.__validate_labels = np.zeros(validate_positive_samples.shape[0] + validate_negative_samples.shape[0], dtype=int)
        self.__validate_labels[: validate_positive_samples.shape[0]] = 1
        
        self.__validate_dataloader = DataLoader(Samples(validate_samples), batch_size=self.__batch_size, shuffle=False)

        self.imbalance_expanding = self.__conf.getHP('imbalance_expanding')
        if self.imbalance_expanding != 'none':
            self.current_imbalance_ratio = 2
            self.current_divider = 2

        if self.__conf.getHP('model') == 'resnetfr':
            self.__refine_extra_iterations = self.__conf.getHP('refine_extra_iterations')
            self.__refine_bpfilter_iterations = self.__conf.getHP('refine_bpfilter_iterations')

            self.__trainF = self.__trainFR
            self.__loss_refine = self.__getLoss(self.__conf.getHP('refine_loss_function'))

            self.__all_train_dataloader = DataLoader(Samples(self.__all_train_samples), batch_size=self.__batch_size, shuffle=False)
            self.__all_train_labels = self.__train_labels

            self.__all_train_weights = self.__train_weights

            if self.__num_train_negative_samples != self.__total_num_train_negative_samples:
                self.__all_train_labels = torch.zeros(self.__num_train_positive_samples + self.__total_num_train_negative_samples, device=self.__device, requires_grad=False)
                self.__all_train_labels[: self.__num_train_positive_samples] = 1

                if self.__conf.getHP('loss_function') == 'ce':
                    self.__all_train_weights = torch.ones(self.__num_train_positive_samples + self.__total_num_train_negative_samples, device=self.__device, requires_grad=False)

                    if self.__conf.getHP('ce_weights') == 'cvpr19':
                        pweight = 1 / self.__num_train_positive_samples
                        nweight = 1 / self.__num_train_negative_samples

                        self.__all_train_weights[: self.__num_train_positive_samples] = 2 * pweight / (pweight + nweight)
                        self.__all_train_weights[self.__num_train_positive_samples: ] = 2 * nweight / (pweight + nweight)
    
            self.__all_train_labels_numpy = self.__all_train_labels.cpu().numpy()
        else:
            self.__trainF = self.__train
        
        self.__optimizer = self.__getOptimizer()
        self.__loss = self.__getLoss()

        self.__initModel()
        
        autograd_hacks.add_hooks(self.model)        

        if self.__conf.getHP('warmup'):
            if self.__conf.getHP('model') != 'resnetfr':
                self.__all_train_dataloader = DataLoader(Samples(self.__all_train_samples), batch_size=self.__batch_size, shuffle=False)
                self.__all_train_labels = self.__train_labels

                if self.__num_train_negative_samples != self.__total_num_train_negative_samples:
                    self.__all_train_labels = torch.zeros(self.__num_train_positive_samples + self.__total_num_train_negative_samples, device=self.__device, requires_grad=False)
                    self.__all_train_labels[: self.__num_train_positive_samples] = 1

                self.__all_train_labels_numpy = self.__all_train_labels.cpu().numpy()

            self.__warmupModel()

        self.sampler = None

        if self.__num_train_negative_samples < self.__total_num_train_negative_samples and self.__conf.getHP('sample_method') == 'latent':
            self.sampler = Sampler(self.__conf)


    def step(self):
        start = timer()

        self.__adjustLR()
        self.__adjustWD()

        train_loss, tfbeta, tprecision, trecall = self.__trainF()

        if type(train_loss) is tuple:
            self.__logger.info('t{:d} in {:.3f}s: lf={:.5f}, lr={:.5f}, f{:.1f}={:.5f}, pre={:.5f}, rec={:.5f}'.format(
                self.iteration, timer() - start, train_loss[0], train_loss[1], self.__conf.getHP('f_beta'), tfbeta, tprecision, trecall))
        else:
            self.__logger.info('t{:d} in {:.3f}s: l={:.5f}, f{:.1f}={:.5f}, pre={:.5f}, rec={:.5f}'.format(
                self.iteration, timer() - start, train_loss, self.__conf.getHP('f_beta'), tfbeta, tprecision, trecall))
        
        start = timer()

        vfbeta, vprecision, vrecall = self.__validate()

        self.__logger.info('v{:d} in {:.3f}s: f{:.1f}={:.5f}, pre={:.5f}, rec={:.5f}'.format(
            self.iteration, timer() - start, self.__conf.getHP('f_beta'), vfbeta, vprecision, vrecall))

        self.iteration += 1

        return train_loss, tfbeta, tprecision, trecall, vfbeta, vprecision, vrecall


    def __train(self) -> None:
        losses = []

        predictions = []
        targets = []

        if self.__num_train_negative_samples != self.__total_num_train_negative_samples:
            start = timer()

            negative_indices = None

            if self.iteration == 0:
                negative_indices = 'random'
            elif self.iteration % self.__conf.getHP('sampling_period') == 0:
                if self.sampler is None:
                    negative_indices = 'random'
                else:
                    negative_indices = self.sampler.sample(self.__getLatent()) + self.__num_train_positive_samples

            if negative_indices is not None:
                if self.imbalance_expanding != 'none':
                    if self.iteration != 0 and self.iteration % self.current_divider == 0:
                        self.current_imbalance_ratio = int(2 * self.current_imbalance_ratio)
                        self.current_divider = int(2 * self.current_divider)

                    current_num_train_negative_samples = int(self.current_imbalance_ratio * self.__num_train_positive_samples)

                    if current_num_train_negative_samples >= self.__num_train_negative_samples:
                        current_num_train_negative_samples = self.__num_train_negative_samples
                else:
                    current_num_train_negative_samples = self.__num_train_negative_samples

                if type(negative_indices) is str and negative_indices == 'random':
                    negative_indices = np.random.permutation(
                        np.arange(self.__num_train_positive_samples, 
                                  self.__num_train_positive_samples + self.__total_num_train_negative_samples)
                        )[: current_num_train_negative_samples]

                    sample_str = 'random'
                else:
                    sample_str = 'latent'

                indices = np.concatenate([np.arange(self.__num_train_positive_samples), negative_indices])

                self.__train_dataloader = DataLoader(SamplesLabelsWeights(self.__all_train_samples[indices], 
                                                                          self.__train_labels[: self.__num_train_positive_samples + current_num_train_negative_samples], 
                                                                          self.__train_weights), 
                                                     batch_size=self.__batch_size, shuffle=True)
            
                self.__logger.info('sample ({:s}) t{:d} in {:.3f}s'.format(sample_str, self.iteration, timer() - start))

        for batch_samples, batch_labels, batch_weights in self.__train_dataloader:
            autograd_hacks.clear_backprops(self.model)

            self.__optimizer.zero_grad()

            batch_predictions = self.model(batch_samples)

            batch_loss = self.__loss(batch_predictions, batch_labels, batch_weights)

            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2, error_if_nonfinite=True)

            autograd_hacks.compute_grad1(self.model)

            print('===========================================')
            print('batch_predictions')
            print(batch_labels)
            print(batch_predictions)
            print(batch_loss)
            # print(batch_loss.grad)

            print('===========================================')
            print('model._ResNet__infer[0].weight')
            print(self.model._ResNet__infer[0].weight.shape, torch.mean(self.model._ResNet__infer[0].weight), torch.mean(self.model._ResNet__infer[0].weight.grad))
            grad1 = torch.squeeze(torch.mean(self.model._ResNet__infer[0].weight.grad1, dim=-1))
            print(torch.mean(grad1[batch_labels == 0]), torch.mean(grad1[batch_labels == 1]), torch.mean(self.model._ResNet__infer[0].weight.grad1))
            print(grad1)

            # print('===========================================')
            # print('model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight')
            # print(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight.shape, torch.mean(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight))
            # if (self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight.grad is not None):
            #     print(torch.mean(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight.grad))
            # else:
            #     print(torch.mean(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight_g.grad), torch.mean(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight_v.grad))

            # print('===========================================')
            # print('model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight')
            # print(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight.shape, torch.mean(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight))
            # if (self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight.grad is not None):
            #     print(torch.mean(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight.grad))
            # else:
            #     print(torch.mean(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight_g.grad), torch.mean(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight_v.grad))

            self.__optimizer.step()

            losses.append(batch_loss.detach().item())
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(batch_labels.clone().detach().cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        predictions = discretize(predictions, self.__threshold)

        targets = np.concatenate(targets, axis=0)

        fbeta, precision, recall = Fbeta(predictions, targets)
        
        return np.mean(losses), fbeta, precision, recall


    def __trainFR(self) -> None:
        train_loss, _, _, _ = self.__train()

        predictions = []

        with torch.no_grad():
            for batch_samples in self.__all_train_dataloader:
                predictions.append(self.model.infer(batch_samples))

        predictions = np.concatenate(predictions, axis=0)

        fbeta, precision, recall = Fbeta(predictions, self.__all_train_labels_numpy)

        filter_samples_indices = np.concatenate((np.asarray(range(self.__num_train_positive_samples)), 
                                                 np.squeeze(np.argwhere(predictions[self.__num_train_positive_samples: ]))))

        if self.__conf.getHP('refine_loss_function') != 'ce':
            self.__loss_refine.reset(p2n=self.__num_train_positive_samples / (np.sum(predictions) - self.__num_train_positive_samples))

        losses = []

        filter_weights = None
    
        filter_dataloader = DataLoader(SamplesLabelsWeights(self.__all_train_samples[filter_samples_indices], 
                                                            self.__all_train_labels[filter_samples_indices], 
                                                            filter_weights), 
                                                batch_size=self.__batch_size, shuffle=True)

        for epoch in range(1, self.__refine_extra_iterations + 1):
            detach = self.__refine_bpfilter_iterations is None or epoch not in self.__refine_bpfilter_iterations

            for batch_samples, batch_labels, batch_weights in filter_dataloader:
                self.__optimizer.zero_grad()

                batch_predictions = self.model(batch_samples, refine=True, detach=detach)

                batch_loss = self.__loss_refine(batch_predictions, batch_labels, batch_weights)

                batch_loss.backward()
                self.__optimizer.step()

                if epoch == self.__refine_extra_iterations:
                    losses.append(batch_loss.detach().item())
        
        return (train_loss, np.mean(losses)), fbeta, precision, recall


    def __getLatent(self) -> np.ndarray:
        latents = []

        for batch_samples in DataLoader(Samples(self.__train_negative_samples), batch_size=self.__batch_size, shuffle=False):
            self.model(batch_samples)
            latents.append(self.model.latent4label.detach().cpu().numpy())

        return np.concatenate(latents, axis=0)


    def __validate(self) -> None:
        predictions = []

        for batch_samples in self.__validate_dataloader:
            predictions.append(self.model.infer(batch_samples))

        predictions = np.concatenate(predictions, axis=0)

        return Fbeta(predictions, self.__validate_labels)


    def predict(self, samples: np.ndarray) -> np.ndarray:
        dim_series = self.__conf.getHP('dim_series')
        num_channels = self.__conf.getHP('num_input_channels')

        samples = np.asarray(samples, dtype=np.float32)

        assert len(samples.shape) == 3 and samples.shape[1] == num_channels and samples.shape[2] == dim_series

        samples = torch.from_numpy(samples).to(self.__device)

        predictions = []

        for batch in DataLoader(samples, batch_size=self.__batch_size, shuffle=False):
            predictions.append(self.model.infer(batch))

        predictions = np.concatenate(predictions, axis=0)

        del samples
        gc.collect()
        torch.cuda.empty_cache()

        return predictions


    def cleanup(self, checkpoint: bool = True):
        self.__logger.info('train finishes by {:d}'.format(self.iteration))

        if checkpoint:
            checkpoint_filepath = self.__conf.getHP('checkpoint_filepath')

            assert checkpoint_filepath is not None and type(checkpoint_filepath) is str 
            assert not os.path.exists(checkpoint_filepath) and Path(checkpoint_filepath).parent.absolute().exists()

            torch.save(self.model.state_dict(), checkpoint_filepath)
 
        del self.__all_train_samples
        del self.__train_dataloader
        
        del self.__validate_dataloader

        del self.model
        del self.__optimizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def __warmupModel(self) -> nn.Module:
        self.__adjustLR(1e-3)

        warmup_loss = BCELoss()
        warmup_epochs = self.__conf.getHP('warmup_epochs')
        
        warmup_samples = torch.cat([
            self.__train_positive_samples, 
            self.__train_negative_samples[np.random.permutation(self.__num_train_negative_samples)][: self.__num_train_positive_samples],
        ])

        warmup_labels = self.__train_labels[: warmup_samples.shape[0]]

        warmup_dataloader = DataLoader(SamplesLabelsWeights(warmup_samples, warmup_labels), batch_size=self.__batch_size, shuffle=True)

        for warmup_epoch in range(warmup_epochs):
            start = timer()

            losses = []

            predictions = []
            targets = []

            for batch_samples, batch_labels, batch_weights in warmup_dataloader:
                autograd_hacks.clear_backprops(self.model)

                self.__optimizer.zero_grad()

                batch_predictions = self.model(batch_samples)

                batch_loss = warmup_loss(batch_predictions, batch_labels, batch_weights)

                batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2, error_if_nonfinite=True)

                autograd_hacks.compute_grad1(self.model)

                print('===========================================')
                print('batch_predictions')
                print(batch_labels)
                print(batch_predictions)
                print(batch_loss)
                # print(batch_loss.grad)

                print('===========================================')
                print('model._ResNet__infer[0].weight')
                print(self.model._ResNet__infer[0].weight.shape, torch.mean(self.model._ResNet__infer[0].weight), torch.mean(self.model._ResNet__infer[0].weight.grad))
                grad1 = torch.squeeze(torch.mean(self.model._ResNet__infer[0].weight.grad1, dim=-1))
                print(torch.mean(grad1[batch_labels == 0]), torch.mean(grad1[batch_labels == 1]), torch.mean(self.model._ResNet__infer[0].weight.grad1))
                print(grad1)

                # print('===========================================')
                # print('model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight')
                # print(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight.shape, torch.mean(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight))
                # if (self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight.grad is not None):
                #     print(torch.mean(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight.grad))
                # else:
                #     print(torch.mean(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight_g.grad), torch.mean(self.model._ResNet__cnn._ResCNN__output._PreActivatedResBlock__residual_link[5].weight_v.grad))

                # print('===========================================')
                # print('model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight')
                # print(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight.shape, torch.mean(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight))
                # if (self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight.grad is not None):
                #     print(torch.mean(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight.grad))
                # else:
                #     print(torch.mean(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight_g.grad), torch.mean(self.model._ResNet__cnn._ResCNN__input._PreActivatedResBlock__residual_link[5].weight_v.grad))

                self.__optimizer.step()

                losses.append(batch_loss.detach().item())
                
                predictions.append(batch_predictions.detach().cpu().numpy())
                targets.append(batch_labels.clone().detach().cpu().numpy())

                if self.__conf.getHP('model') == 'resnetfr':
                    self.__optimizer.zero_grad()
                    
                    batch_predictions = self.model(batch_samples, refine=True, detach=True)

                    batch_loss = warmup_loss(batch_predictions, batch_labels, batch_weights)

                    batch_loss.backward()
                    self.__optimizer.step()
        
            predictions = np.concatenate(predictions, axis=0)
            predictions = discretize(predictions, self.__threshold)

            targets = np.concatenate(targets, axis=0)

            fbeta, precision, recall = Fbeta(predictions, targets)

            self.__logger.info('w{:d} in {:.3f}s: l={:.5f}, f{:.1f}={:.5f}, pre={:.5f}, rec={:.5f}'.format(
                warmup_epoch, timer() - start, np.mean(losses), self.__conf.getHP('f_beta'), fbeta, precision, recall))

        start = timer()
        predictions = []

        with torch.no_grad():
            for batch_samples in self.__all_train_dataloader:
                predictions.append(self.model(batch_samples).detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        predictions = discretize(predictions, self.__threshold)

        fbeta, precision, recall = Fbeta(predictions, self.__all_train_labels_numpy)

        self.__logger.info('t-1 in {:.3f}s: f{:.1f}={:.5f}, pre={:.5f}, rec={:.5f}'.format(
            timer() - start, self.__conf.getHP('f_beta'), fbeta, precision, recall))

        start = timer()
        predictions = []

        with torch.no_grad():
            for batch_samples in self.__validate_dataloader:
                predictions.append(self.model(batch_samples).detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        predictions = discretize(predictions, self.__threshold)

        fbeta, precision, recall = Fbeta(predictions, self.__validate_labels)

        self.__logger.info('v-1 in {:.3f}s: f{:.1f}={:.5f}, pre={:.5f}, rec={:.5f}'.format(
            timer() - start, self.__conf.getHP('f_beta'), fbeta, precision, recall))
        

    def __initModel(self) -> nn.Module:
        if self.__conf.getHP('model_init') == 'lsuv':
            lsuv_size = self.__conf.getHP('lsuv_size')

            if lsuv_size < 0:
                lsuv_samples = torch.cat([
                    self.__train_positive_samples, 
                    self.__train_negative_samples[np.random.permutation(self.__num_train_negative_samples)][: self.__num_train_positive_samples],
                ])
            else:
                lsuv_samples = torch.cat([
                    self.__train_positive_samples[np.random.permutation(self.__num_train_positive_samples)][: int(lsuv_size / 2)], 
                    self.__train_negative_samples[np.random.permutation(self.__num_train_negative_samples)][: int(lsuv_size / 2)],
                ])
            
            lsuv_samples = lsuv_samples[np.random.permutation(lsuv_samples.shape[0])]

            self.model = LSUVinit(self.model, 
                                  lsuv_samples,
                                  needed_mean=self.__conf.getHP('lsuv_mean'), 
                                  needed_std=self.__conf.getHP('lsuv_std'), 
                                  std_tol=self.__conf.getHP('lsuv_std_tol'), 
                                  max_attempts=self.__conf.getHP('lsuv_maxiter'), 
                                  do_orthonorm=self.__conf.getHP('lsuv_ortho'),
                                  logger=self.__logger)


    def __getLoss(self, loss_function: str = None) -> nn.Module:
        if loss_function is None:
            loss_function = self.__conf.getHP('loss_function')

        if loss_function.endswith('ce'):
            return BCELoss()
        elif loss_function.endswith('sf'):
            float32_epsilon = self.__conf.getHP('float32_epsilon')

            beta = self.__conf.getHP('f_beta')
            p2n = self.__num_train_positive_samples / self.__num_train_negative_samples

            alpha = self.__conf.getHP('f_alpha')

            if loss_function == 'sf':
                assert np.abs(alpha - 1) < float32_epsilon

            return FbetaLoss(beta=beta, alpha=alpha, p2n=p2n, epsilon=float32_epsilon)
            # return FbetaLoss(beta=beta, alpha=alpha, p2n=p2n, epsilon=float32_epsilon, reduction='none')
        else:
            raise ValueError('invalid loss function: {:s}'.format(loss_function))


    def __getOptimizer(self) -> optim.Optimizer:
        if self.__conf.getHP('optim_type') == 'sgd':
            if self.__conf.getHP('lr_mode') == 'fix':
                initial_lr = self.__conf.getHP('lr_cons')
            else:
                initial_lr = self.__conf.getHP('lr_max')

            if self.__conf.getHP('wd_mode') == 'fix':
                initial_wd = self.__conf.getHP('wd_cons')
            else:
                initial_wd = self.__conf.getHP('wd_min')

            momentum = self.__conf.getHP('momentum')

            return optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=initial_wd)

        raise ValueError('invalid optimizer name: {:s}'.format(self.__conf.getHP('optim_type')))


    def __adjustLR(self, new_lr = None) -> None:
        if new_lr == None:
            for param_group in self.__optimizer.param_groups:
                current_lr = param_group['lr']
                break
            
            new_lr = current_lr

            if self.__conf.getHP('lr_mode') == 'linear':
                lr_max = self.__conf.getHP('lr_max')
                lr_min = self.__conf.getHP('lr_min')

                new_lr = lr_max - self.iteration * (lr_max - lr_min) / self.__max_epoch
            elif self.__conf.getHP('lr_mode') == 'exponentiallyhalve':
                lr_max = self.__conf.getHP('lr_max')
                lr_min = self.__conf.getHP('lr_min')
                
                for i in range(1, 11):
                    if (self.__max_epoch - self.iteration) * (2 ** i) == self.__max_epoch:
                        new_lr = lr_max / (10 ** i)
                        break

                if new_lr < lr_min:
                    new_lr = lr_min
            elif self.__conf.getHP('lr_mode') == 'exponentially':
                lr_max = self.__conf.getHP('lr_max')
                lr_min = self.__conf.getHP('lr_min')
                lr_k = self.__conf.getHP('lr_everyk')
                lr_ebase = self.__conf.getHP('lr_ebase')

                lr_e = int(np.floor(self.iteration / lr_k))
                new_lr = lr_max * (lr_ebase ** lr_e)

                if new_lr < lr_min:
                    new_lr = lr_min
            elif self.__conf.getHP('lr_mode') == 'plateauhalve':
                raise ValueError('plateauhalve is not yet supported')

        for param_group in self.__optimizer.param_groups:
            param_group['lr'] = new_lr


    def __adjustWD(self):
        for param_group in self.__optimizer.param_groups:
            current_wd = param_group['weight_decay']
            break
        
        new_wd = current_wd

        if self.__conf.getHP('wd_mode') == 'linear':
            wd_max = self.__conf.getHP('wd_max')
            wd_min = self.__conf.getHP('wd_min')

            new_wd = wd_min + self.iteration * (wd_max - wd_min) / self.__max_epoch

        for param_group in self.__optimizer.param_groups:
            param_group['weight_decay'] = new_wd
