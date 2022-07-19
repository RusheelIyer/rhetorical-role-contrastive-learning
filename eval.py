import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
import numpy as np

from utils import tensor_dict_to_gpu, tensor_dict_to_cpu


def calc_classification_metrics(y_true, y_predicted, labels):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='micro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='weighted')
    per_label_precision, per_label_recall, per_label_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average=None, labels=labels)

    acc = accuracy_score(y_true, y_predicted)

    class_report = classification_report(y_true, y_predicted, digits=4)
    confusion_abs = confusion_matrix(y_true, y_predicted, labels=labels)
    # normalize confusion matrix
    confusion = np.around(confusion_abs.astype('float') / confusion_abs.sum(axis=1)[:, np.newaxis] * 100, 2)
    return {"acc": acc,
            "macro-f1": macro_f1,
            "macro-precision": macro_precision,
            "macro-recall": macro_recall,
            "micro-f1": micro_f1,
            "micro-precision": micro_precision,
            "micro-recall": micro_recall,
            "weighted-f1": weighted_f1,
            "weighted-precision": weighted_precision,
            "weighted-recall": weighted_recall,
            "labels": labels,
            "per-label-f1": per_label_f1.tolist(),
            "per-label-precision": per_label_precision.tolist(),
            "per-label-recall": per_label_recall.tolist(),
            "confusion_abs": confusion_abs.tolist()
            }, \
           confusion.tolist(), \
           class_report


# Store the embeddings after a given layer in the activation dict with the
# corresponding layer name as a key
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def eval_model(model, eval_batches, device, task, contrastive, memory_bank, memory_bank_labels):
    
    # register a hook to receive activations after sentence_lstm layer
    model.sentence_lstm.register_forward_hook(get_activation('sentence_lstm'))
    model.eval()

    true_labels = []
    labels_dict={}
    predicted_labels = []
    docwise_predicted_labels=[]
    docwise_true_labels = []
    doc_name_list = []
    sentence_embeddings = []

    with torch.no_grad():
        for batch in eval_batches:
            # move tensor to gpu
            tensor_dict_to_gpu(batch, device)

            if batch["task"] != task.task_name:
                continue

            if (contrastive):
                output, sentence_embeddings_batch, features = model(batch=batch)
            else:
                output, sentence_embeddings_batch = model(batch=batch)

            # get the batches sentence embeddings
            # sentence_embeddings_batch = activation['sentence_lstm']

            knn_predicted_labels, _ = kNN(memory_bank, memory_bank_labels, features, batch["label_ids"])
            print(type(batch["label_ids"]))
            print(batch["label_ids"].shape)
            print(predicted_labels.shape)
            true_labels_batch, predicted_labels_batch = \
            clear_and_map_padded_values(batch["label_ids"].view(-1), knn_predicted_labels.view(-1), task.labels)
                #clear_and_map_padded_values(batch["label_ids"].view(-1), output["predicted_label"].view(-1), task.labels)

            docwise_true_labels.append(true_labels_batch)
            docwise_predicted_labels.append(predicted_labels_batch)
            doc_name_list.append(batch['doc_name'][0])

            true_labels.extend(true_labels_batch)
            predicted_labels.extend(predicted_labels_batch)

            sentence_embeddings.extend(sentence_embeddings_batch)

            tensor_dict_to_cpu(batch)
    labels_dict['y_true']=true_labels
    labels_dict['y_predicted'] = predicted_labels
    labels_dict['labels'] = task.labels
    labels_dict['docwise_y_true'] = docwise_true_labels
    labels_dict['docwise_y_predicted'] = docwise_predicted_labels
    labels_dict['doc_names'] = doc_name_list
    metrics, confusion, class_report = \
        calc_classification_metrics(y_true=true_labels, y_predicted=predicted_labels, labels=task.labels)

    # Save the sentence embeddings to external file
    torch.save(sentence_embeddings, 'datasets/embeddings.pt')
    
    return metrics, confusion, labels_dict, class_report

'''
Params:
    true_labels, predicted_labels: label IDs
    labels: array of labels with IDs
'''
def clear_and_map_padded_values(true_labels, predicted_labels, labels):
    assert len(true_labels) == len(predicted_labels)
    cleared_predicted = []
    cleared_true = []
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        # filter masked labels (0)
        if true_label > 0:
            cleared_true.append(labels[true_label])
            cleared_predicted.append(labels[predicted_label])
    return cleared_true, cleared_predicted

def kNN(memory_bank, memory_bank_labels, features, feature_labels):

    pred_labels = torch.zeros_like(feature_labels)

    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            sentence = features[i][j].unsqueeze(0)
            dist = torch.norm(memory_bank - sentence, dim=2, p=None)

            _, knn_indices = dist.topk(10, largest=False)
            pred_labels[i][j] = get_majority_label(memory_bank_labels[i][knn_indices])

    total_labels = feature_labels.shape[1]

    return pred_labels, (torch.eq(pred_labels, feature_labels).sum()/total_labels)*100

def get_majority_label(labels):
    return torch.mode(labels).values.item()