import torch
# from torch_geometric.data import Data

def gen_distance_matrix(tsp_coordinates):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(tsp_coordinates)
    # [n_nodes, n_nodes]
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    # 初始化自身节点的距离
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances
    
def gen_pyg_data(tsp_coordinates, k_sparse):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    n_nodes = len(tsp_coordinates)
    # 获取距离矩阵 [n_nodes, n_nodes]
    distances = gen_distance_matrix(tsp_coordinates)
    # topk个最小值和索引 [n_nodes, k_sparse]
    topk_values, topk_indices = torch.topk(distances, 
                                           k=k_sparse, 
                                           dim=1, largest=False)
    # 边索引 [2, n_nodes*topk]
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse), # [n_nodes*topk]
        torch.flatten(topk_indices)  # [n_nodes*topk]
        ])
    # 边权重 [n_nodes, 1]
    edge_attr = topk_values.reshape(-1, 1)
    # pyg数据，节点表征（坐标），边索引，边权重（距离）
    pyg_data = Data(x=tsp_coordinates, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

def load_val_dataset(n_node, k_sparse, device):
    val_list = []
    val_tensor = torch.load(f'../data/tsp/valDataset-{n_node}.pt')
    # [instance_num, node_num, coordinate]
    for instance in val_tensor:
        instance = instance.to(device)
        # 每个instance的数据
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse)
        val_list.append((data, distances))
    return val_list

def load_test_dataset(n_node, k_sparse, device):
    val_list = []
    val_tensor = torch.load(f'../data/tsp/testDataset-{n_node}.pt')
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse)
        val_list.append((data, distances))
    return val_list

if __name__ == "__main__":
    load_val_dataset(20, 10, 'cpu')
