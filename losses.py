import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    
    def __init__(self, dim_in, feat_dim, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    """
    Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    def forward(self, batch, features):

        # device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        documents, sentences, _ = batch["input_ids"].shape
        # labels = batch["label_ids"].to(device)
        labels = batch["label_ids"]

        features=self.head(features)

        if labels is not None:
            """mask = torch.eq(labels, labels.T).float()
            logits_mask = torch.scatter(
                torch.ones_like(mask).to(device),
                1,
                torch.arange(sentences * 1).view(-1, 1).to(device),
                0
            )
            mask = (mask * logits_mask).to(device)"""

            mask = torch.eq(labels, labels.T).float()
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(sentences * 1).view(-1, 1),
                0
            )
            mask = (mask * logits_mask)
        else:
            #mask = mask.float().to(device)
            mask = mask.float()

        # norms = features.norm(dim=1).view(-1,1).to(device)
        norms = features.norm(dim=1).view(-1,1)
        norms = torch.matmul(norms, norms.T)

        """cos_similarity_matrix = torch.div(
            torch.div(
                torch.matmul(features, features.T),
                norms
                ),
            0.07).to(device)"""
        cos_similarity_matrix = torch.div(
            torch.div(
                torch.matmul(features, features.T),
                norms
                ),
            0.07)

        #numerator = torch.exp(cos_similarity_matrix * mask).to(device)
        numerator = torch.exp(cos_similarity_matrix * mask)

        denom = (torch.exp(cos_similarity_matrix)*logits_mask).sum(1)
        #denom = denom.view(-1,1).repeat(1,sentences).to(device)
        denom = denom.view(-1,1).repeat(1,sentences)

        #log_prob = (torch.log(torch.div(numerator, denom))*mask).to(device)
        #num_positives = mask.sum(1).view(-1,1).repeat(1,sentences).to(device)
        #loss = torch.mul(torch.div(-1, num_positives),log_prob).nansum().to(device)
        log_prob = (torch.log(torch.div(numerator, denom))*mask)
        num_positives = mask.sum(1).view(-1,1).repeat(1,sentences)
        loss = torch.mul(torch.div(-1, num_positives),log_prob).nansum()

        return loss