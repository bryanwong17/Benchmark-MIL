import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, mid_dim, out_dim=1, flatten=False, dropout=0.):
        super(AttentionPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.flatten = flatten

        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.out_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.squeeze(0)

        H = x.view(x.shape[0], -1)  # d * L
        H = torch.transpose(H, 1, 0)  # L * d

        A = self.attention(H)  # L * K
        A = torch.transpose(A, 1, 0)  # K * L
        A = F.softmax(A, dim=1)  # softmax over L
        A = self.dropout(A)

        M = torch.mm(A, H)  # Kxd

        if self.flatten:
            M = M.flatten()
        return M.unsqueeze(0)