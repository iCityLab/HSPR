import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def single_non_zero_to_one(lst):
    indices = [i for i, x in enumerate(lst) if x != 0]
    if len(indices) == 1:
        # print(lst)
        return [1 if i == indices[0] else 0 for i in range(len(lst))]
    else:
        return lst




def calculated_reg_proximity(c_matrix):
    percentile_95 = torch.quantile(c_matrix, 0.95, dim=1)

    # Replace outliers
    # Use broadcasting to ensure the shape of percentile_95 matches the count
    winsorized_data = torch.where(c_matrix > percentile_95.unsqueeze(1), percentile_95.unsqueeze(1), c_matrix)

    winsorized_data_np = winsorized_data.numpy()

    # Create MinMaxScaler object, set specific range
    scaler = MinMaxScaler(feature_range=(0, 0.95))
    normalized_data = []
    for line in winsorized_data_np:
        pro = scaler.fit_transform(line.reshape(-1, 1)).flatten()
        normalized_data.append(pro)
    normalized_data = np.vstack(normalized_data)
    conn_pro = np.around(np.array(normalized_data), 4) + np.eye(31, dtype=np.float32)
    return conn_pro


def calculated_obj_proximity(o_matrix):
    percentile_95 = torch.quantile(o_matrix, 0.95, dim=1)

    winsorized_data = torch.where(o_matrix > percentile_95.unsqueeze(1), percentile_95.unsqueeze(1), o_matrix)

    winsorized_data_np = winsorized_data.numpy()

    scaler = MinMaxScaler(feature_range=(0, 0.95))
    normalized_data = []
    for line in winsorized_data_np:
        pro = scaler.fit_transform(line.reshape(-1, 1)).flatten()
        normalized_data.append(pro)
    normalized_data = np.vstack(normalized_data)
    conn_pro = np.around(np.array(normalized_data), 4) + np.eye(1600, dtype=np.float32)
    # print(conn_pro.shape)
    return conn_pro
