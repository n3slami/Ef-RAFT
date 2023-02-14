import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """ Positional encoding. """
    def __init__(self, num_hiddens, dropout=0.0, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(1, max_len + 1, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.sinx = torch.sin(X)
        self.cosx = torch.cos(X)
        self.P[:, :, 0::2] = self.sinx
        self.P[:, :, 1::2] = self.cosx

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

    def to_relatvive(self, X):
        all_sin = torch.clone(X[:, :, 0::2])
        all_cos = torch.clone(X[:, :, 1::2])
        sinx = self.sinx[:all_sin.shape[1], :].to(X.device)
        cosx = self.cosx[:all_cos.shape[1], :].to(X.device) 
        X[:, :, 0::2] = all_sin * cosx - all_cos * sinx
        X[:, :, 1::2] = all_cos * cosx + all_sin * sinx
        return X

class CoordinateAttention(nn.Module):
    def __init__(self, feature_size=128, enc_size=128, heads=4, bias=True, dropout=0.0):
        super(CoordinateAttention, self).__init__()
        self.feature_size = feature_size
        self.enc_size = enc_size
        self.att = nn.MultiheadAttention(feature_size, heads, dropout=dropout, bias=bias,
                                            batch_first=True)
        self.pos_enc = PositionalEncoding(feature_size)
    
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-1] == self.feature_size
        dev = x.device
        # Convert feature map to attention friendly format
        row_x = x.view(-1, x.shape[-2], self.feature_size).to(dev)
        col_x = torch.permute(x, (0, 2, 1, 3)).to(dev)
        col_x = col_x.reshape(-1, col_x.shape[-2], self.feature_size)
        
        # Get positional encoding
        row_vals = self.pos_enc(torch.zeros_like(row_x).to(dev))
        col_vals = self.pos_enc(torch.zeros_like(col_x).to(dev))

        # Calculate output and convert to relative
        row_res = self.pos_enc.to_relatvive(self.att(row_x, row_x, row_vals)[0])
        row_res = row_res.view(*x.shape)[:, :, :, :self.enc_size]
        col_res = self.pos_enc.to_relatvive(self.att(col_x, col_x, col_vals)[0])
        col_res = col_res.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3])[:, :, :, :self.enc_size]
        return torch.cat((x, row_res, col_res), dim=-1)

ca = CoordinateAttention(feature_size=8, enc_size=4)
pe = PositionalEncoding(num_hiddens=8)
y = torch.zeros(1, 16, 8)
res = pe(y)
print(res[:, :, 0::2])
print(res[:, :, 1::2])
res = pe.to_relatvive(res)
print(res[:, :, 0::2])
print(res[:, :, 1::2])

print("########################################")

x = torch.rand(2, 4, 4, 8)
print(x)
y = ca(x)
print(x.shape, y.shape)
print(y)