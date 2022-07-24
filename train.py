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

                    if (memory_bank == None):
                        memory_bank = features
                        memory_bank_labels = batch["label_ids"]
                    else:
                        memory_bank = torch.cat((memory_bank, features), dim=1).detach()
                        memory_bank_labels = torch.cat((memory_bank_labels, batch["label_ids"]), dim=1).detach()
                elif self.config['task_type'] == 'proto_sim':
                    output, sentence_embeddings_encoded, prototypes = model(
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
                    contrastive_loss = self.ConLossFunc(batch, features, memory_bank, memory_bank_labels)
                    #contrastive_loss = self.SupCon(memory_bank, memory_bank_labels, features)
                    
                    cl_beta = 1
                    loss = (classification_loss) + (cl_beta*contrastive_loss)
                elif self.config['task_type'] == 'proto_sim':
                    protosim_loss = self.ConLossFunc(batch, sentence_embeddings_encoded, prototypes)

                    lambda_3 = 0.5
                    loss = (lambda_3*classification_loss) + protosim_loss
                else:
                    loss = classification_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # move batch to cpu again to save gpu memory
                tensor_dict_to_cpu(batch)

                if batch_num % 100 == 0:
                    self.result_writer.log(f"Loss in fold {fold_num}, epoch {epoch}, batch {batch_num}: {loss.item()}")

            train_duration = time.time() - epoch_start

            epoch_scheduler.step()

            # evaluate model
            results={}
            self.result_writer.log(f'evaluating model...')
            dev_metrics, dev_confusion,labels_dict, _ = eval_model(model, dev_batches, self.device, self.task, self.config['contrastive'], memory_bank, memory_bank_labels)
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
                test_metrics, test_confusion,labels_dict,_ = eval_model(model, test_batches, self.device, self.task, self.config['contrastive'], memory_bank, memory_bank_labels)
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