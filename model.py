import torch
from torch.nn import functional as F
from layers import GraphConvolutionBS, JKNetBlock, InceptionGCNBlock, ResGCNBlock, DenseJKNetBlock


class EvoGCN(torch.nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(self, args, record, in_channels, out_channels, drop, hidden_dim):
        super(EvoGCN, self).__init__()
        # 参数
        self.record = record
        self.dropout = drop
        self.hidden_dim = hidden_dim
        # 网络层
        self.ingconv = GraphConvolutionBS(in_channels, self.hidden_dim)
        self.midgconvs = torch.nn.ModuleList()
        for i in range(len(record)):
            if record[i][0] == 0:
                self.midgconvs.append(ResGCNBlock(self.hidden_dim, self.hidden_dim, record[i][1] * args.res_block_len, dropout=drop))
            elif record[i][0] == 1:
                self.midgconvs.append(JKNetBlock(self.hidden_dim, self.hidden_dim, record[i][1] * args.dense_block_len, dropout=drop))
            elif record[i][0] == 2:
                self.midgconvs.append(InceptionGCNBlock(self.hidden_dim, self.hidden_dim, record[i][1] * args.incep_block_len, dropout=drop))
            elif record[i][0] == 3:
                self.midgconvs.append(DenseJKNetBlock(self.hidden_dim, self.hidden_dim, record[i][1] * args.dense_block_len, dropout=drop))
            else:
                raise Exception('error')
        self.outgconvs = GraphConvolutionBS(self.hidden_dim, out_channels)

    def forward(self, x, adj, edge_weight=None):
        x = F.dropout(self.ingconv(x, adj), self.dropout, training=self.training)  # 降维要改
        for i in range(len(self.record)):
            x = self.midgconvs[i](x, adj, edge_weight)
        x = self.outgconvs(x, adj)  # 降维要改
        x = F.log_softmax(x, dim=1)
        return x

