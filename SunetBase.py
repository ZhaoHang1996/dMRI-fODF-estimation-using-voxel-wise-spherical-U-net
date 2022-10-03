import scipy.io as sio 
import numpy as np
import torch
from torch.utils.data import Dataset


neigh_id_path = r"C:\Users\31758\Desktop\SphericalUNetPackage-main\sphericalunet\utils\neigh_indices\adj_mat_order_"
# neigh_id_path = "/data/zh/EXP_save/sunet/neigh_indices/adj_mat_order_"

class Database(Dataset):
    def __init__(self, indata, label):

        self.indata = torch.from_numpy(indata)
        self.label = torch.from_numpy(label)

    def __getitem__(self,index):
        return self.indata[index], self.label[index]

    def __len__(self):
        return len(self.indata)

def Get_neighs_order(rotated=0):
    neigh_orders_163842 = get_neighs_order(neigh_id_path +'163842_rotated_' + str(rotated) + '.mat')
    neigh_orders_40962 = get_neighs_order(neigh_id_path +'40962_rotated_' + str(rotated) + '.mat')
    neigh_orders_10242 = get_neighs_order(neigh_id_path +'10242_rotated_' + str(rotated) + '.mat')
    neigh_orders_2562 = get_neighs_order(neigh_id_path +'2562_rotated_' + str(rotated) + '.mat')
    neigh_orders_642 = get_neighs_order(neigh_id_path +'642_rotated_' + str(rotated) + '.mat')
    neigh_orders_162 = get_neighs_order(neigh_id_path +'162_rotated_' + str(rotated) + '.mat')
    neigh_orders_42 = get_neighs_order(neigh_id_path +'42_rotated_' + str(rotated) + '.mat')
    neigh_orders_12 = get_neighs_order(neigh_id_path +'12_rotated_' + str(rotated) + '.mat')
    
    return neigh_orders_163842, neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12
  
def get_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders

def Get_upconv_index(rotated=0):
    
    upconv_top_index_163842, upconv_down_index_163842 = get_upconv_index(neigh_id_path+'163842_rotated_' + str(rotated) + '.mat')
    upconv_top_index_40962, upconv_down_index_40962 = get_upconv_index(neigh_id_path+'40962_rotated_' + str(rotated) + '.mat')
    upconv_top_index_10242, upconv_down_index_10242 = get_upconv_index(neigh_id_path+'10242_rotated_' + str(rotated) + '.mat')
    upconv_top_index_2562, upconv_down_index_2562 = get_upconv_index(neigh_id_path+'2562_rotated_' + str(rotated) + '.mat')
    upconv_top_index_642, upconv_down_index_642 = get_upconv_index(neigh_id_path+'642_rotated_' + str(rotated) + '.mat')
    upconv_top_index_162, upconv_down_index_162 = get_upconv_index(neigh_id_path+'162_rotated_' + str(rotated) + '.mat')
    
    return upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 

def get_upconv_index(order_path):  
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index