import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    """
    Compute loss for model.
    Args:
        batch: ground truth
        features: hidden vector of shape [sentences, 128 (feat_dim)].
    Returns:
        A loss scalar.
    """
    def forward(self, batch, features):

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        documents, sentences, _ = batch["input_ids"].shape
        labels = batch["label_ids"].to(device)

        if labels is not None:
            mask = torch.eq(labels, labels.T).float()
            logits_mask = torch.scatter(
                torch.ones_like(mask).to(device),
                1,
                torch.arange(sentences * 1).view(-1, 1).to(device),
                0
            )
            mask = (mask * logits_mask).to(device)

        else:
            mask = mask.float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_count = features.shape[1]

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                0.07
            )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)
        
        # avoid div by 0 NaN for labels without positive samples
        denom = mask.sum(1)
        denom = denom ** (denom!=0)
        mean_log_prob_pos /= denom

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, documents).sum()

        return loss

class SupConLossMemory(nn.Module):
    
    def __init__(self, temperature=0.07):
        super(SupConLossMemory, self).__init__()
        self.temperature = temperature
        self.SupCon = SupConLoss()

    def forward(self, batch, features, memory_bank, memory_bank_labels):

        contr_loss = self.SupCon(batch, features)

        feature_labels = batch["label_ids"]
        _, knn_loss = self.kNN(memory_bank, memory_bank_labels, features, feature_labels)

        return contr_loss + knn_loss

    def kNN(self, memory_bank, memory_bank_labels, features, feature_labels):

        pred_labels = torch.zeros_like(feature_labels)

        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                sentence = features[i][j].unsqueeze(0)
                dist = torch.norm(memory_bank - sentence, dim=2, p=None)

                _, knn_indices = dist.topk(10, largest=False)
                pred_labels[i][j] = self.get_majority_label(memory_bank_labels[i][knn_indices])

        total_labels = feature_labels.shape[1]

        return pred_labels, (torch.eq(pred_labels, feature_labels).sum()/total_labels)*100

    def get_majority_label(self, labels):
        return torch.mode(labels).values.item()

    """
    Compute loss for model.
    Args:
        memory_bank: [all_sentences, 128 (feat_dim)]
        features: hidden vector of shape [sentences, 128 (feat_dim)].
    Returns:
        A loss scalar.
    
    def forward(self, memory_bank, memory_bank_labels, features):

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        sentences = memory_bank.shape[1]
        
        labels = memory_bank_labels.to(device)
        memory_bank.to(device)

        contrast_feature = torch.cat(torch.unbind(memory_bank, dim=1), dim=0)
        contrast_count = memory_bank.shape[1]

        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_count = features.shape[1]

        memory_bank.to('cpu') # Free up GPU

        if labels is not None:
            mask = torch.eq(labels, labels.T).float()
            logits_mask = torch.scatter(
                torch.ones_like(mask).to(device),
                1,
                torch.arange(sentences * 1).view(-1, 1).to(device),
                0
            )[-anchor_count:]
            mask = (mask[-anchor_count:] * logits_mask).to(device)

        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                0.07
            )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)
        
        # avoid div by 0 NaN for labels without positive samples
        denom = mask.sum(1)
        denom = denom ** (denom!=0)
        mean_log_prob_pos /= denom

        # loss
        loss = -1 * mean_log_prob_pos

        return loss.sum()
    """

class ProtoSimLoss(nn.Module):
    
    def __init__(self, lambda_1 = 1, lambda_2 = 0.5):
        super(ProtoSimLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    """
    Compute loss for model.
    Args:
        batch: ground truth
        features: hidden vector of shape [sentences, 128 (feat_dim)].
    Returns:
        A loss scalar.
    """
    def forward(self, batch, features, prototypes):

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        documents, sentences, _ = batch["input_ids"].shape
        labels = batch["label_ids"].to(device)

        cp_loss = self.get_contrastive_loss(device, features, labels, sentences)
        cluster_loss = self.get_cluster_loss(device, features, labels, sentences, prototypes)

        return (self.lambda_1*cp_loss) + (self.lambda_2*cluster_loss)

    def get_contrastive_loss(self, device, features, labels, sentences):

        if labels is not None:
            mask = torch.eq(labels, labels.T).float()
            logits_mask = torch.scatter(
                torch.ones_like(mask).to(device),
                1,
                torch.arange(sentences * 1).view(-1, 1).to(device),
                0
            )
            mask = (mask * logits_mask).to(device)

        else:
            mask = mask.float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_count = features.shape[1]

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
                1,
                1 + torch.exp(torch.matmul(anchor_feature,contrast_feature.T))
            )

        pos_num = torch.exp(anchor_dot_contrast*mask)*logits_mask
        neg_denom = (torch.exp(anchor_dot_contrast*(1-mask))*logits_mask).sum(1)

        return (-1/sentences**2)*torch.div(pos_num, neg_denom).sum()

    def get_cluster_loss(self, device, features, labels, sentences, prototypes):

        old_mask = torch.eq(labels, labels.T).float().to(device)
        
        dists_z2s = torch.div(1, 1 + torch.exp(torch.matmul(prototypes, anchor_feature.T)))
        dists_s2z = torch.div(1, 1 + torch.exp(torch.matmul(anchor_feature, prototypes.T)))

        pos_dists = ((torch.log(dists_s2z)* old_mask).sum(1)/old_mask.sum(1)).view(-1,1)
        neg_dists = torch.log(1-dists_z2s)*(1-old_mask)

        ls2z = ((pos_dists + neg_dists)*(1-old_mask)).sum()

        counts = old_mask.sum(1).tile(sentences).view(sentences,sentences)
        neg_dists_z_ = torch.log(1 - (dists_s2z*(1-old_mask)/counts))
        pos_dists_z_ = pos_dists/old_mask.sum(1)*(1-old_mask)

        ls2z_ = (pos_dists_z_ + neg_dists).sum()

        return (-1/sentences**2)*(ls2z + ls2z_)