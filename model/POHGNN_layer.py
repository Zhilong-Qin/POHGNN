import torch
import torch.nn as nn
import numpy as np
from ipdb import set_trace
import torch.nn.functional as F
from model.POHGNN_base import POHGNN_ctr_ntype_specific


class POHGNN_nc_mb_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 in_dim,
                 out_dim,
                 attn_drop=0.5):
        super(POHGNN_nc_mb_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim


        self.ctr_ntype_layer = POHGNN_ctr_ntype_specific(num_metapaths, etypes_list, in_dim)

        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        # 得到的这个h，就是GTN所需要的
        features, type_mask, adj_matrixes, feature_idxes = inputs

        h = torch.zeros(type_mask.shape[0], self.in_dim, device=features.device)
        # 最后聚合的h
        h_temp, temps = self.ctr_ntype_layer((features, type_mask, adj_matrixes, feature_idxes))
        h[np.where(type_mask == 0)[0]] = h_temp
        temp1, temp2 ,temp3= temps
        h[np.where(type_mask == 1)[0]] = temp1.float()
        h[np.where(type_mask == 2)[0]] = temp2.float()
        h[np.where(type_mask == 3)[0]] = temp3.float()
        ####  一个1024*3的全连接层，得到的最后的输出结果
        h_fc = self.fc(h)
        return h_fc, h


class POHGNN_nc_mb(nn.Module):
    def __init__(self, num_layers,num_metapaths, etypes_list, feats_dim_list, hidden_dim, out_dim, dropout_rate=0.5, adjD = None, count = None, count2= None):
        super(POHGNN_nc_mb, self).__init__()
        self.hidden_dim = hidden_dim
        self.adjD = adjD
        self.num_layers=num_layers

        # node centrality encoding
        all_degree = list(set(count))
        all_degree.sort()
        self.degree_index = np.where(np.repeat(count[:,None], len(all_degree), axis=1) == np.repeat(np.array(all_degree)[None,:], adjD.shape[0], axis=0))[1]
        self.degree_embedding = torch.eye(len(all_degree))

        all_degree2 = list(set(count2))
        all_degree2.sort()
        self.degree_index2 = np.where(np.repeat(count2[:,None], len(all_degree2), axis=1) == np.repeat(np.array(all_degree2)[None,:], adjD.shape[0], axis=0))[1]
        self.degree_embedding2 = torch.eye(len(all_degree2))

        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        self.fc_list_c = nn.ModuleList([nn.Linear(len(all_degree)+len(all_degree2), hidden_dim, bias=True) for feats_dim in feats_dim_list])
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.7)

        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for fc in self.fc_list_c:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # SHGNN_nc_mb layers
        # self.layer1 = SHGNN_nc_mb_layer(num_metapaths, etypes_list, hidden_dim*2, out_dim,attn_drop=dropout_rate)

        self.layers = nn.ModuleList()
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(
                POHGNN_nc_mb_layer(num_metapaths, etypes_list, hidden_dim * 2, hidden_dim * 2, attn_drop=dropout_rate))
        # output projection layer
        self.layers.append(POHGNN_nc_mb_layer(num_metapaths, etypes_list, hidden_dim * 2, out_dim, attn_drop=dropout_rate))

    def forward(self, inputs, target_node_indices):
        features_list, type_mask, adj_matrixes, feature_idxes = inputs
        device = features_list[0].device
        # type-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=device)
        transformed_features_c = torch.zeros(type_mask.shape[0], self.hidden_dim, device=device)

        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            # features_list=features_list.float()
            # self.degree_embedding = self.degree_embedding.float()
            # self.degree_embedding2 = self.degree_embedding2.float()

            transformed_features[node_indices] = self.leaky_relu(fc(features_list[i].float()))
            transformed_features_c[node_indices] = self.leaky_relu(self.fc_list_c[i](torch.cat([self.degree_embedding[self.degree_index].float(), self.degree_embedding2[self.degree_index2].float()], axis=1)[node_indices].to(device)))

        transformed_features = self.feat_drop(torch.cat([transformed_features, transformed_features_c], axis=1))
        transformed_features = transformed_features.half()


        #change
        for l in range(self.num_layers - 1):
            h, _ = self.layers[l]((transformed_features, type_mask, adj_matrixes, feature_idxes))
            h = F.elu(h)
            # 从这里改
        logits, h = self.layers[-1]((h, type_mask, adj_matrixes, feature_idxes))

        # logits, h = self.layer1((transformed_features, type_mask, adj_matrixes, feature_idxes))
        return logits[target_node_indices], h[target_node_indices]
