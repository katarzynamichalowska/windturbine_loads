import numpy as np

def load_preprocessed_data(data_dir, add_info_to_X=False):
    data = np.load(data_dir, allow_pickle=True)

    X, y, scaler_X, scaler_y = data["X"], data["y"], data["scaler_X"], data["scaler_Y"]
    columns_X, columns_Y = data["columns_X"], data["columns_Y"]
    info, info_columns = data["info"], data["info_columns"]

    if add_info_to_X:
        info_brd = np.repeat(np.expand_dims(info, axis=1), X.shape[1], axis=1)
        X = np.concatenate((X, info_brd), axis=2)

    return X, y, scaler_X, scaler_y, columns_X, columns_Y, info, info_columns
