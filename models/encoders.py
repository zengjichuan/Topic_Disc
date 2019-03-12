import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, emb_dim, emb_num, config):
        super(CNN, self).__init__()
        if config.emb_mat is None:
            V = emb_num
            D = emb_dim
        else:
            V,D = config.emb_mat.shape
        C = config.cls_num
        Ci = 1
        Co = config.kernel_num
        Ks = config.kernel_sizes
        self.static = config.static_emb
        self.embed = nn.Embedding(V, D, padding_idx=config.pad_wid)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class MultiFC(nn.Module):
    """
    Applies fully connected layers to an input vector
    """
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=0, short_cut=False):
        super(MultiFC, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.short_cut = short_cut
        if num_hidden_layers == 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc_layers = nn.ModuleList()
            self.fc_input = nn.Linear(input_size, hidden_size)
            self.fc_output = nn.Linear(hidden_size, output_size)
            if short_cut:
                self.es = nn.Linear(input_size, output_size)
            for i in range(1, self.num_hidden_layers):
                self.fc_layers.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, input_var):
        if self.num_hidden_layers == 0:
            out = self.fc(input_var)
        else:
            x = F.tanh(self.fc_input(input_var))
            for i in range(1, self.num_hidden_layers):
                x = F.tanh(self.fc_layers[i](x))
            out = self.fc_output(x)
        if self.short_cut:
            out = out + self.es(input_var)
        return out