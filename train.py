from prettytable import PrettyTable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from losses import SupConLoss, SupConLossMemory, ProtoSimLoss

import torch
import torch.nn.functional as F
import random
import json
import time
import models
import numpy as np
import os

from eval import eval_model
from utils import tensor_dict_to_gpu, tensor_dict_to_cpu, ResultWriter, get_num_model_parameters, print_model_parameters
from task import Task, Fold
import gc
import copy

class SentenceClassificationTrainer:
    '''Trainer for baseline model and also for Sequantial Transfer Learning. '''
    def __init__(self, device, config, task: Task, result_writer:ResultWriter):
        self.device = device
        self.config = config
        self.result_writer = result_writer
        self.cur_result = dict()
        self.cur_result["task"] = task.task_name
        self.cur_result["config"] = config

        self.labels = task.labels
        self.task = task
        if (config["task_type"] == 'contrastive'):
            self.ConLossFunc = SupConLoss()
        elif (config["task_type"] == 'memory'):
            self.ConLossFunc = SupConLossMemory()
        elif (config["task_type"] == 'proto_sim'):
            self.ConLossFunc = ProtoSimLoss()

    def write_results(self, fold_num, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion):
        self.cur_result["fold"] = fold_num
        self.cur_result["epoch"] = epoch
        self.cur_result["train_duration"] = train_duration
        self.cur_result["dev_metrics"] = dev_metrics
        self.cur_result["dev_confusion"] = dev_confusion
        self.cur_result["test_metrics"] = test_metrics
        self.cur_result["test_confusion"] = test_confusion

        self.result_writer.write(json.dumps(self.cur_result))

    """
    Add 10 samples per label ID for the memory bank
    """
    def get_memory_features(self, features, labels, memory_bank, memory_bank_labels, bank_type='random', num_samples=10):
        
        label_ids = len(self.task.labels)
        docs, _, feat_dim = features.shape

        memory_bank_new = torch.zeros(docs, num_samples, feat_dim)
        memory_bank_labels_new = torch.zeros(docs, num_samples)

        if bank_type == 'random':
            for i in range(docs):
                for label_id in range(1, label_ids, 1):
                    indices = (labels[i] == label_id).nonzero(as_tuple=True)[0].tolist()
                    
                    if (len(indices) > 0):
                        if (len(indices) <= num_samples):
                            sample_idxs = random.choices(indices, k=num_samples)
                        else:
                            sample_idxs = random.sample(indices, num_samples)
                        
                        memory_bank_new[i] = features[i][sample_idxs]
                        memory_bank_labels_new[i] = labels[i][sample_idxs]
        else:
            for i in range(docs):
                label_features = {}
                for label_id in range(1, label_ids, 1):
                    indices = (labels[i] == label_id).nonzero(as_tuple=True)[0].tolist()
                    
                    if len(indices) > 0:
                        memory_bank_new[i] = self.get_closest_features(features[i][indices], num_samples)
                        memory_bank_labels_new[i] = labels[i][indices[0]].repeat(num_samples)

        if memory_bank is None:
            return memory_bank_new.detach(), memory_bank_labels_new.detach()
        else:
            return torch.cat((memory_bank, memory_bank_new), dim=1).detach(), torch.cat((memory_bank_labels, memory_bank_labels_new), dim=1).detach()

    """
    Calculate centroid for feature set and return closest 10 samples to the centroid.
    Functional use case: expects features belonging to same class.
    """
    def get_closest_features(self, features, num_samples):

        if features.shape[0] < num_samples:
            sample_idxs = random.choices(range(features.shape[0]), k=num_samples)
            return features[sample_idxs]

        if num_samples == features.shape[0]:
            return features

        mean = torch.mean(features, dim=0)
        centroid = torch.ones_like(features)*mean

        pdist = torch.nn.PairwiseDistance(p=2)
        distances = pdist(features, centroid)
        min_idxs = torch.topk(distances, num_samples, largest=False).indices

        return features[min_idxs]
        
    def run_training_for_fold(self, fold_num, fold: Fold, initial_model=None, return_best_model=False):

        self.result_writer.log(f'device: {self.device}')

        train_batches, dev_batches, test_batches = fold.train, fold.dev, fold.test

        self.result_writer.log(f"fold: {fold_num}")
        self.result_writer.log(f"train batches: {len(train_batches)}")
        self.result_writer.log(f"dev batches: {len(dev_batches)}")
        self.result_writer.log(f"test batches: {len(test_batches)}")

        # instantiate model per reflection
        if initial_model is None:
            model = getattr(models, self.config["model"])(self.config, [self.task])
        else:
            self.result_writer.log("Loading weights from initial model....")
            model = copy.deepcopy(initial_model)
            # for transfer learning do not transfer the output layer
            model.reinit_output_layer([self.task], self.config)

        self.result_writer.log("Model: " + model.__class__.__name__)
        self.cur_result["model"] = model.__class__.__name__

        model.to(self.device)

        max_train_epochs = self.config["max_epochs"]
        lr = self.config["lr"]
        max_grad_norm = 1.0

        self.result_writer.log(f"Number of model parameters: {get_num_model_parameters(model)}")
        self.result_writer.log(f"Number of model parameters bert: {get_num_model_parameters(model.bert)}")
        self.result_writer.log(f"Number of model parameters word_lstm: {get_num_model_parameters(model.word_lstm)}")
        self.result_writer.log(f"Number of model parameters attention_pooling: {get_num_model_parameters(model.attention_pooling)}")
        self.result_writer.log(f"Number of model parameters sentence_lstm: {get_num_model_parameters(model.sentence_lstm)}")
        self.result_writer.log(f"Number of model parameters crf: {get_num_model_parameters(model.crf)}")
        print_model_parameters(model)

        # for feature based training use Adam optimizer with lr decay after each epoch (see Jin et al. Paper)
        optimizer = Adam(model.parameters(), lr=lr)
        epoch_scheduler = StepLR(optimizer, step_size=1, gamma=self.config["lr_epoch_decay"])

        best_dev_result = 0.0
        early_stopping_counter = 0
        epoch = 0
        early_stopping = self.config["early_stopping"]
        best_model = None

        #memory_bank = torch.zeros(total_train_sentences, 2*config["word_lstm_hs"])

        optimizer.zero_grad()
        while epoch < max_train_epochs and early_stopping_counter < early_stopping:
            epoch_start = time.time()

            self.result_writer.log(f'training model for fold {fold_num} in epoch {epoch} ...')

            random.shuffle(train_batches)
            
            memory_bank = None
            memory_bank_labels = None
            # train model
            model.train()
            for batch_num, batch in enumerate(train_batches):
                # move tensor to gpu
                tensor_dict_to_gpu(batch, self.device)

                if self.config['task_type'] == 'contrastive':
                    output, sentence_embeddings_encoded, features = model(
                        batch=batch,
                        labels=batch["label_ids"]
                    )
                elif self.config['task_type'] == 'memory':
                    output, sentence_embeddings_encoded, features = model(
                        batch=batch,
                        labels=batch["label_ids"]
                    )

                    memory_bank, memory_bank_labels = self.get_memory_features(features, batch["label_ids"], memory_bank, memory_bank_labels, bank_type='centroid')

                elif self.config['task_type'] == 'proto_sim':
                    output, sentence_embeddings_encoded, features, prototypes = model(
                        batch=batch,
                        labels=batch["label_ids"]
                    )
                else:
                    output, sentence_embeddings_encoded = model(
                        batch=batch,
                        labels=batch["label_ids"]
                    )

                classification_loss = output["loss"].sum()
                if self.config['task_type'] == 'contrastive':
                    contrastive_loss = self.ConLossFunc(batch, features)
                    cl_beta = 1
                    loss = (classification_loss) + (cl_beta*contrastive_loss)
                elif self.config['task_type'] == 'memory':
                    contrastive_loss = self.ConLossFunc(batch, features, memory_bank, memory_bank_labels)
                    cl_beta = 1
                    loss = (classification_loss) + (cl_beta*contrastive_loss)
                elif self.config['task_type'] == 'proto_sim':
                    protosim_loss = self.ConLossFunc(batch, features, prototypes)

                    lambda_3 = 0.5
                    loss = (lambda_3*classification_loss) + protosim_loss
                else:
                    loss = classification_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # clear cache and move batch to cpu again to save gpu memory
                torch.cuda.empty_cache()
                tensor_dict_to_cpu(batch)

                if batch_num % 100 == 0:
                    self.result_writer.log(f"Loss in fold {fold_num}, epoch {epoch}, batch {batch_num}: {loss.item()}")

            train_duration = time.time() - epoch_start

            epoch_scheduler.step()

            # evaluate model
            results={}
            self.result_writer.log(f'evaluating model...')
            dev_metrics, dev_confusion,labels_dict, _ = eval_model(model, dev_batches, self.device, self.task, self.config['task_type'])
            results['dev_metrics']=dev_metrics
            results['dev_confusion'] = dev_confusion
            results['labels_dict'] = labels_dict
            results['classification_report']=_


            if dev_metrics[self.task.dev_metric] > best_dev_result:
                if return_best_model:
                    best_model = copy.deepcopy(model)
                best_dev_result = dev_metrics[self.task.dev_metric]
                early_stopping_counter = 0
                self.result_writer.log(f"New best dev {self.task.dev_metric} {best_dev_result}!")
                results={}
                test_metrics, test_confusion,labels_dict,_ = eval_model(model, test_batches, self.device, self.task, self.config['task_type'])
                results['dev_metrics']=dev_metrics
                results['dev_confusion'] = dev_confusion
                results['labels_dict'] = labels_dict
                results['classification_report']=_


                self.write_results(fold_num, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion)
                self.result_writer.log(
                    f'*** fold: {fold_num},  epoch: {epoch}, train duration: {train_duration}, dev {self.task.dev_metric}: {dev_metrics[self.task.dev_metric]}, test weighted-F1: {test_metrics["weighted-f1"]}, test macro-F1: {test_metrics["macro-f1"]}, test accuracy: {test_metrics["acc"]}')
            else:
                early_stopping_counter += 1
                self.result_writer.log(f'fold: {fold_num}, epoch: {epoch}, train duration: {train_duration}, dev {self.task.dev_metric}: {dev_metrics[self.task.dev_metric]}')

            epoch += 1
        return best_model