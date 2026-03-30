import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cpu'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i, z_j):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
        z_i, z_j: (B, D) projections from two augmented views.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)  # (2B, D)
        
        # Normalize vectors for cosine similarity
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(N, dtype=torch.bool).to(self.device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        target = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ], dim=0).to(self.device)
        
        loss = F.cross_entropy(sim_matrix, target)
        return loss

class MultilabelFocalLoss(nn.Module):
    """Multi-label focal loss for BCEWithLogits."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super(MultilabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):  # multi-hot targets
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma * bce)
        return focal.mean()
