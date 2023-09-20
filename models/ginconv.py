import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import os
import math
from collections import OrderedDict

import numpy as np
# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, embed_dim2=100,output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd =Linear(dim, 128)
        self.fc_end = Linear(128, 128)

        # 1D convolution on protein sequence
        self.embedding_drug = nn.Embedding(600, 100)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        #self.embedding_xt2 = nn.Embedding(num_features_xt + 1, embed_dim2)
        self.conv_xt_1 = nn.Conv1d(in_channels=1200, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)
        self.drugGlobal1 = nn.Linear(78, 32)
        self.drugGlobal2 = nn.Linear(32, 32)
        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)
        self.dout = nn.Linear(128, 128)
        self.pout = nn.Linear(128, 128)        # n_output = 1 for regression task
        self.Linear1 = nn.Linear(128, 128)
        self.Linear2 = nn.Linear(128, 128)
        self.meanPool2d = nn.AvgPool2d((1, 2), stride=1)
        self.pmeanPool2d = nn.AvgPool2d((1,2),stride=1)
        self.pLinear1 = nn.Linear(128, 128)
        self.pLinear2 = nn.Linear(128, 128)
        self.conv2d = nn.Conv2d(1, 32, (3,32))
        # self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.LinearAdd2 = nn.Linear(1200, 1200)
        self.Sigmoid = nn.Sigmoid()
        self.LeakyD =nn.LeakyReLU()
        self.LeakyP = nn.LeakyReLU()

        #78→24
        self.drugmaxpool = nn.MaxPool2d([1,54])

        #分子距离图
        self.contact_cov = nn.Conv2d(1, 32, (1200,1))
        self.bn6 = torch.nn.BatchNorm1d(dim)
        self.contact_ReLU = nn.ReLU()
        self.contact_cov2 = nn.Conv2d(1, 32, (1, 32))
        self.linear_p4_1 = nn.Linear(32, 32)
        self.linear_p4_2 = nn.Linear(32, 32)
        self.FC_Pout = nn.Linear(32, 128)

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        input = input.transpose(axis, len(input_size) - 1)
        trans_size = input.size()
        return F.softmax((input.contiguous().view(-1, trans_size[-1])), dim=1).view(*trans_size).transpose(axis,
                                                                                                           len(input_size) - 1)

    def forward(self, data,epoch):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        p4 = data.p4
        psite = data.psite
        site =[]

        #drug-------------------------------------------------
        for key in psite:
            key = ''.join(key)
            key = key.split('A')
            key1=[]
            for i in key:
                i = int(i)
                i = math.ceil(i/10)
                key1.append(i)

            site.append(key1)

        x_0 = F.relu(self.conv1(x, edge_index))
        x_0 = self.bn1(x_0)
        x_0 = F.relu(self.conv2(x_0, edge_index))
        x_0 = self.bn2(x_0)
        x_0 = F.relu(self.conv3(x_0, edge_index))
        x_0 = self.bn3(x_0)
        x_0 = F.relu(self.conv4(x_0, edge_index))
        x_0 = self.bn4(x_0)
        x_0 = F.relu(self.conv5(x_0, edge_index))
        x_0 = self.bn5(x_0)
        x_0 = global_add_pool(x_0, batch)
        x_0 = F.relu(self.fc1_xd(x_0))
        x_0 = F.dropout(x_0, p=0.2, training=self.training)
        #x = 512,128

        x_1 = self.drugGlobal1(x)
        x_1 = self.drugGlobal2(x_1)
        x_1 = global_add_pool(x_1, batch)
        x_1 = F.relu(self.fc1_xd(x_1))
        x_1 = F.dropout(x_1, p=0.2, training=self.training)


        d1Learn = self.Linear1(x_0)
        d1Learn = torch.relu(d1Learn)
        d2Learn = self.Linear2(x_1)
        d2Learn = torch.relu(d2Learn)
        x_0 = x_0.reshape(x_0.shape[0], x_0.shape[1], 1)
        x_1 = x_1.reshape(x_1.shape[0], x_1.shape[1], 1)
        Mprotein = torch.cat((x_0, x_1), 2)
        outputDrug = self.meanPool2d(Mprotein)
        outputDrug = torch.sigmoid_(outputDrug)
        oneOutputDrug = 1 - outputDrug
        outputDrug = outputDrug.squeeze()
        oneOutputDrug = oneOutputDrug.squeeze()
        FWa = d1Learn * outputDrug
        F1_Wa = d2Learn * oneOutputDrug
        x_0 = x_0.squeeze()
        x_1 = x_1.squeeze()
        Dout = FWa + x_0 + F1_Wa + x_1

        #Pretion 分子距离图-------------------------------------------------
        contac_path = 'data/' + 'davis' + '/pconsc4'
        contact_list=[]
        # 1
        for key in p4:
            contact_file = os.path.join(contac_path, key[0] + '.npy')
            contact_map = np.load(contact_file)
            if contact_map.shape[0] <= 1200 :
                num = 1200 - contact_map.shape[0]
                pad_width1 = ((0, num), (0, num))
                contact_map = np.pad(contact_map, pad_width=pad_width1, mode='constant', constant_values=0)
            else:
                contact_map = contact_map[0:1200,0:1200]
            contact_list.append(contact_map)
        contact_list = np.asarray(contact_list)
        contact_list = torch.from_numpy(contact_list)
        contact_list = torch.unsqueeze(contact_list, 1)
        contact_list = contact_list.permute(0, 1, 3, 2)
        contact_list = self.contact_cov(contact_list.cuda())
        contact_list = contact_list.squeeze()
        contact_list_1 = self.bn6(contact_list)
        contact_list_1 = self.contact_ReLU(contact_list_1)
        contact_list_1 = torch.unsqueeze(contact_list_1, 1)
        contact_list_1 = contact_list_1.permute(0, 1, 3, 2)
        contact_list_1 = self.contact_cov2(contact_list_1.cuda())
        contact_list_1 = contact_list_1.squeeze()
        contact = contact_list_1+contact_list
        contact_1 = contact.permute(0, 2, 1)
        contact_1 = self.linear_p4_1(contact_1)
        contact_1 = torch.tanh(contact_1)
        contact_1 = self.linear_p4_2(contact_1)
        seq_att = self.softmax(contact_1).transpose(1,2)
        p_0 = seq_att*contact
        p_0 = p_0.transpose(1, 2)
        p_0 = self.FC_Pout(p_0)
        p_0 = self.conv_xt_1(p_0)
        p_0 = p_0.view(-1, 32 * 121)
        p_0 = self.fc1_xt(p_0)
        # 1
        #pretion  p2-------------------------------------------
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        p_1 = conv_xt.view(-1, 32 * 121)
        p_1 = self.fc1_xt(p_1)
        # Pretion attention-------------------------------------------------

        p1Learn = self.pLinear1(p_0)#1
        p1Learn = torch.relu(p1Learn)#1
        p2Learn = self.pLinear2(p_1)
        p2Learn = torch.relu(p2Learn)

        p_0 = p_0.reshape(p_0.shape[0], p_0.shape[1], 1)#1
        p_1 = p_1.reshape(p_1.shape[0], p_1.shape[1], 1)
        Mprotein = torch.cat((p_0, p_1), 2)#1
        outputP = self.meanPool2d(Mprotein)
        outputP = torch.sigmoid_(outputP)
        oneOutputP = 1 - outputP
        outputP = outputP.squeeze()
        oneOutputP = oneOutputP.squeeze()
        FWap = p1Learn * outputP#1
        F1_Wap = p2Learn * oneOutputP
        p_0 = p_0.squeeze()#1
        p_1 = p_1.squeeze()
        Pout = FWap + p_0 + F1_Wap + p_1#1
        #Pout =  p_1


        # 全连接层 pout 16.1200.32     dout 16.32-------------------------------------------------
        Dout = self.dout(Dout)
        Pout = self.pout(Pout)
        Dout = self.LeakyD(Dout)
        Pout = self.LeakyP(Pout)
        Dout = Dout.permute(1,0)
        alph = torch.mm(Pout, Dout)
        alph = torch.tanh(alph)
        alphdrug = torch.tanh(torch.sum(alph, 1))
        alphprotein = torch.tanh(torch.sum(alph, 1))

        alphdrug = alphdrug.unsqueeze(1).repeat(1, 1, Dout.shape[0])
        alphprotein = alphprotein.unsqueeze(1).repeat(1, 1, Pout.shape[1])
        drug_feature = torch.mul(alphdrug.permute(0, 2, 1), Dout)
        protein_feature = torch.mul(alphprotein, Pout)

        
        protein_feature = protein_feature.squeeze()

        if epoch% 20 == 0:
            sorted, indices = torch.sort(protein_feature,descending=True)
            indices = indices[:,0:15]
            indices = indices.tolist()
            tf_csv=[]
            for i in range(len(site)):
                for j in range(len(indices)):
                    if i == j:
                        c_out = indices[j]+site[i]
                        if len(c_out) != len(set(c_out)):
                            tf_csv.append(0)
                        else:
                            tf_csv.append(1)
               
            with open('results/epoch.txt', 'a') as f:
                f.write('\t  epoch = \t' + str(epoch) + '\t  list = \t' + str(tf_csv) + '\n')




        drug_feature = drug_feature.squeeze()

        drug_feature = self.fc_end(drug_feature.permute(1,0))
        drug_feature = F.dropout(drug_feature, p=0.2, training=self.training)


        xc = torch.cat((drug_feature, protein_feature), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
