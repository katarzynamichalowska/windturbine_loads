import numpy as np
import os
import warnings
from modules.dir_functions import read_parameters, read_params_module_as_dict
#from modules.wavelets import wavelet_length, transform_data_wavelet, reconstruct_wavelet
import sys
sys.path.insert(0, "../")


"""--------------------   PROCESSING (GENERAL)   --------------------"""




def preprocess_data_for_model(X, y, scaler_X, scaler_y, col_names_X, col_names_Y, info, params):

    # Subset variables
    X = subset_variables(data=X, col_names_data=col_names_X, col_subset=params["X_vars"], axis=2)
    y = subset_variables(data=y, col_names_data=col_names_Y, col_subset=params["Y_vars"], axis=2)
    scaler_X = subset_variables(data=scaler_X, col_names_data=col_names_X, col_subset=params["X_vars"], axis=0)
    scaler_y = subset_variables(data=scaler_y, col_names_data=col_names_Y, col_subset=params["Y_vars"], axis=0)

    # Reshape the timeseries to multiple t_len subseries
    X = reshape_series_to_subseries(X, params["t_len"])
    y = reshape_series_to_subseries(y, params["t_len"])
    info = np.repeat(info, int(X.shape[0]/info.shape[0]), axis=0)

    return X, y, scaler_X, scaler_y, info


def update_params_with_default(params):
    """
    Updates params with default values if missing.
    """
    # Default values for incomplete params.txt
    default_values = read_params_module_as_dict('params_default')

    if ('XT_SCALER' not in params) and ('T_SCALER' in params):
    # Change all T_SCALER to XT_SCALER
        params.update({'XT_SCALER':params['T_SCALER']})

    # Update dictionary with default values if missing
    [params.update({key:value}) for key, value in default_values.items() if key not in params]
    
    return params

import numpy as np

def reshape_series(X, y=None, t_len=None, random_start=False, one_ts_per_simulation=False):
    """
    reshape_subseries() over all elements.
    """
    nr_instances = X.shape[0]
    nr_timesteps = X.shape[1]
    nr_X_features = X.shape[2]
    nr_Y_features = y.shape[2]
    
    if random_start and one_ts_per_simulation:
        start_indices = np.random.randint(0, nr_timesteps - t_len + 1, size=nr_instances)
    elif random_start and not one_ts_per_simulation:
        start_indices = np.random.randint(0, t_len + 1, size=nr_instances)
    else:
        start_indices = np.zeros(nr_instances, dtype=int)

    X_subseries = np.zeros((nr_instances, int(nr_timesteps - (nr_timesteps%t_len)-t_len), nr_X_features))
    if y is not None:
        y_subseries = np.zeros((nr_instances, int(nr_timesteps - (nr_timesteps%t_len)-t_len), nr_Y_features))

    # Iterate over each subseries and slice accordingly
    if one_ts_per_simulation:
        for i in range(nr_instances):
            start_idx = start_indices[i]
            end_idx = start_idx + t_len
            
            X_subseries[i] = X[i, start_idx:end_idx]
            if y is not None:
                y_subseries[i] = y[i, start_idx:end_idx]
    else:
        for i in range(nr_instances):
            start_idx = start_indices[i] 
            end_idx = nr_timesteps - (nr_timesteps%t_len) - (t_len-start_idx) 
            X_subseries[i] = X[i, start_idx:end_idx]
            #X_subseries[i] = X_subseries[i].reshape(-1, nr_subinstances, t_len, nr_X_features)
            if y is not None:
                y_subseries[i] = y[i, start_idx:end_idx]#.reshape(-1, nr_subinstances, t_len, nr_Y_features)
            
    if y is None:
        y_subseries = None

    return X_subseries, y_subseries


def reshape_subseries(X, y, t_len, random_start=False):
    """
    Reshape the data into subseries of predefined length.
    """
    if random_start:
        X, y = reshape_subseries_random_start(X, y, t_len)
    else:
        X, y = reshape_subseries_fixed(X, y, t_len)
    return X, y

def reshape_subseries_fixed(X, y, t_len):
    """
    Reshape the data into subseries of predefined length.
    """
    nr_points = int(np.floor(X.shape[0]/t_len)*t_len)
    X = X[:nr_points]
    y = y[:nr_points]
    X = X.reshape(-1, t_len, X.shape[1])
    y = y.reshape(-1, t_len, y.shape[1])
    return X, y

def reshape_subseries_random_start(X, y, t_len):
    """
    Reshape the data into subseries of predefined length, starting from a random nr for each sample.
    """

    nr_points_total = X.shape[0] 
    start = np.random.randint(0, t_len)
    nr_points_left = nr_points_total - start 
    end = int(np.floor(nr_points_left/t_len)*t_len)+start 
    X = X[start:end]
    y = y[start:end]
    print(y.shape)
    X = X.reshape(-1, t_len, X.shape[1])
    y = y.reshape(-1, t_len, y.shape[1])
    return X, y

def reshape_series_to_subseries(X, t_len):
    X = X.reshape(int(X.shape[0]*X.shape[1]/t_len), t_len, X.shape[2])
    return X

def read_model_params(modelname, folder="../../models", filename="params.txt"):
    """
    Returns:
        @ data_processing_params: a dictionary of parameters in lowercase to be used as kwargs 
                                  in preprocess_data() and postprocess_data()
    """

    params_path = os.path.join(os.path.join(folder, modelname), filename)
    params_read = read_parameters(params_path)
    params_read = dict([(key.upper(),value) for key, value in params_read.items()]) #Keys to uppercase
    params_updated = update_params_with_default(params_read)
    data_processing_params = dict([(k.lower(), v) for k,v in zip(params_updated.keys(), params_updated.values())])

    return params_updated, data_processing_params

def subset_variables(data, col_names_data, col_subset, axis=0):

    if col_subset is not None:
        data_idx = [i for i,c in enumerate(col_names_data) if (c in col_subset)]
        data = data.take(indices=data_idx, axis=axis)

    return data


def subset_by_simulation_conditions(X, Y, params, info, info_columns, in_sample=True):
    if X.shape[0] != info.shape[0]:
        raise ValueError(f"X and info have different number of samples: {X.shape[0]} and {info.shape[0]}")
    for p in ["U", "TI", "D", "SH", "DIR"]:
        if params[f"range_{p}"] is not None:
            if not isinstance(params[f"range_{p}"], list):
                raise TypeError(f"range_{p} should be a list, but got {type(params[f'range_{p}'])}. Received: {params[f'range_{p}']}")

            nr_samples_0 = X.shape[0]
            if in_sample:
                idx = np.where((info[:, list(info_columns).index(p)] >= params[f"range_{p}"][0]) & (info[:, list(info_columns).index(p)] <= params[f"range_{p}"][1]))[0]
            else:
                idx = np.where((info[:, list(info_columns).index(p)] < params[f"range_{p}"][0]) | (info[:, list(info_columns).index(p)] > params[f"range_{p}"][1]))[0]
            info = info[idx]
            X,Y = X[idx], Y[idx]
            nr_samples_1 = X.shape[0]
            print(f"Subset by {p} range: {params[f'range_{p}']}")
            print(f"Number of samples before: {nr_samples_0}, after: {nr_samples_1}")
    return X, Y, info


def preprocess_data_from_params(X, xt, y, modelname, folder="../../models", **fixed_params):
    """
    Wrapper for preprocess_data() which reads parameters from logs (params.txt of a given model)

    Params:
        @ fixed_params: preprocessing parameters with a fixed value (not from file)
    """
 
    params_updated, preprocessing_params = read_model_params(modelname, folder=folder)

    # Update with fixed_params if needed
    [preprocessing_params.update({key:value}) for key, value in fixed_params.items()]
    [params_updated.update({key.upper():value}) for key, value in fixed_params.items()]

    return params_updated, preprocess_data(X, xt, y, **preprocessing_params)

def append_g_u_by_u(g_u, u):
    """
    Appends u0 to g_u.
    """
    g_u_temp = g_u.reshape(g_u.shape[0], int(g_u.shape[1]/u.shape[1]), u.shape[1])
    u_temp = u.reshape(u.shape[0], 1, u.shape[1])
    g_u = np.concatenate([u_temp, g_u_temp], axis=1)
    g_u = g_u.reshape(g_u.shape[0], g_u.shape[1]*g_u.shape[2])
    
    return g_u

def preprocess_data(u, xt, g_u, resample_i=None, nr_timesteps=None, train_perc=0.8, u_scaler=None, xt_scaler=None, g_u_scaler=None, 
                    trunk_rnn=False, t1_as_u0=True, t1_as_zeros=False, wt_input=False, wt_output=False,
                    wt_method='db12', wt_mode='smooth', split_outputs=False, x_2d=False, same_scale_in_out=False, 
                    residual=False, add_u0_to_gu=True, scaling_coef_bool=False, target_integ=1, bias=0.2, **kwargs):
    """
    Data preprocessing: 
        - Resamples u and g_u to a lower resolution,
        - Transforms trunk input for batching,
        - Tranforms data into wavelets,
        - Splits the data into training and testing,
        - Scales the data.
    """

    x_len = u.shape[1]
    t_len = int(g_u.shape[1]/x_len)

    if add_u0_to_gu:
        g_u = append_g_u_by_u(g_u, u)
        t_len += 1
        xt = make_xt(x_len=x_len, t_len=t_len)
        xt[:,1] -= 1

    if residual:
        g_u_temp = g_u.reshape(g_u.shape[0], int(g_u.shape[1]/u.shape[1]), u.shape[1])
        g_u_temp[:,1:,:] = g_u_temp[:,1:,:] - g_u_temp[:,:-1,:]
        g_u_temp[:,0,:] = g_u_temp[:,0,:] - u
        g_u = g_u_temp.reshape(g_u.shape[0], g_u.shape[1])

    if x_2d:
        # TODO: not compatible with resampling
        xt = make_xt(x_len, t_len, x_2d=x_2d)

    if (resample_i is not None) and (resample_i != "None"):
        g_u, xt = resample_g_xt(g_u, xt, int(resample_i))
        t_len = int(t_len/resample_i)
        print(f"Output resampled by i={resample_i}.")

    if (nr_timesteps is not None) and (nr_timesteps != "None"):
        if nr_timesteps >= t_len:
            raise ValueError(f"nr_timesteps={nr_timesteps} is larger than number of timestesteps in the data ({t_len})")
        g_u = g_u[:, :nr_timesteps*x_len]
        t_len = int(g_u.shape[1]/x_len)
        xt = make_xt(x_len, t_len, x_2d=x_2d)
        print(f"Output trimmed to {nr_timesteps} timesteps.")


    u_train, g_u_train, u_test, g_u_test, xt_train, xt_test = train_test_split(X=u,
                                                                               y=g_u,
                                                                               xt=xt,
                                                                               train_perc=train_perc,
                                                                               batch_xt=trunk_rnn)
    
    # Transform into wavelets.
    # If (wt_input==False) and (wt_output==False) returns the original data
    u_train_trans, xt_train_trans, g_u_train_trans = transform_data_wavelet(u=u_train, xt=xt_train, g_u=g_u_train, method=wt_method, mode=wt_mode,
                                                                            concat_u=True, concat_g_u=(not split_outputs),
                                                                            transform_input=wt_input, transform_output=wt_output)

    u_test_trans, xt_test_trans, g_u_test_trans = transform_data_wavelet(u=u_test, xt=xt_test, g_u=g_u_test, method=wt_method, mode=wt_mode,
                                                                        concat_u=True, concat_g_u=(not split_outputs),
                                                                        transform_input=wt_input, transform_output=wt_output)
    if wt_output:
        wt_len = wavelet_length(x_len, method=wt_method, mode=wt_mode)

        if split_outputs:
            xt = make_xt(wt_len*2, t_len)
        else:
            xt = make_xt(wt_len, t_len)
    # Scale data. 
    # If (scaler is None) returns the original data
    if train_perc > 0.0:
        u_train_trans, xt_train_trans, g_u_train_trans, u_scaler, xt_scaler, g_u_scaler = scale_data(u=u_train_trans,
                                                                                                    xt=xt_train_trans,
                                                                                                    g_u=g_u_train_trans,
                                                                                                    u_scaler=u_scaler,
                                                                                                    xt_scaler=xt_scaler,
                                                                                                    g_u_scaler=g_u_scaler)
    u_test_trans, xt_test_trans, g_u_test_trans, _, _, _ = scale_data(u=u_test_trans,
                                                                      xt=xt_test_trans,
                                                                      g_u=g_u_test_trans,
                                                                      u_scaler=u_scaler,
                                                                      xt_scaler=xt_scaler,
                                                                      g_u_scaler=g_u_scaler)
    
    if same_scale_in_out and (wt_input==False) and (wt_output==False) and (scaling_coef_bool==False):
            g_u_scaler = u_scaler
            g_u_train_trans, _ = scaling(g_u_train, scaler=u_scaler)
            g_u_test_trans, _ = scaling(g_u_test, scaler=u_scaler)

    if scaling_coef_bool:
        coef_train = scaling_coef(u_train_trans, target_integ=target_integ, axis=1, bias=bias)                          # Coefficient to scale input: shape (5000,)
        coef_test = scaling_coef(u_test_trans, target_integ=target_integ, axis=1, bias=bias)                          # Coefficient to scale input: shape (5000,)
        #g_u_train_trans = scale_by_coef(g_u_train_trans, coef_train)  
        #g_u_test_trans = scale_by_coef(g_u_test_trans, coef_test)
    else:
        coef_train, coef_test = None, None



    return dict({
        'u_train': u_train,
        'u_test': u_test,
        'g_u_train': g_u_train,
        'g_u_test': g_u_test,
        'u_train_trans': u_train_trans,
        'u_test_trans': u_test_trans,
        'g_u_train_trans': g_u_train_trans,
        'g_u_test_trans': g_u_test_trans,
        'xt_train': xt_train,
        'xt_test': xt_test,
        'xt_train_trans': xt_train_trans,
        'xt_test_trans': xt_test_trans,
        'u_scaler': u_scaler,
        'g_u_scaler': g_u_scaler,
        'xt_scaler': xt_scaler,
        'wt_input': wt_input,
        'wt_output': wt_output,
        'wt_method': wt_method,
        'wt_mode': wt_mode,
        'x_len': x_len,
        't_len': t_len,
        'coef_train': coef_train,
        'coef_test': coef_test,
        })

def postprocess_data(f, scaler=None, wavelet=False, wt_method="db12", wt_mode="smooth", data_len=50, 
                     split_output=False, **kwargs):
    """
    Data postprocessing:
    - Unscales the data
    - Converts wavelets to raw data

    Params:
        @ f: function or a list of functions that should be preprocessed in the same way
    """

    f_post = list()

    if not isinstance(f, list):
        f = [f]

    for f_i in f:

        if scaler is not None:
            f_i = inverse_scaling(z=f_i, scaler=scaler)
        
        if wavelet:
            f_i = reconstruct_wavelet(f_i, method=wt_method, mode=wt_mode, data_len=data_len)
        
        f_i = reshape_3d_2d(f_i)
        f_post.append(f_i)
    
    if len(f_post)==1:
        f_post = f_post[0]
    
    return f_post



def postprocess_g_u_from_params(f, modelname, folder="../../models", scaler=None, data_len=50):
    """
    Wrapper for preprocess_data() which reads parameters from logs (params.txt of a given model)
    """

    params, data_processing_params = read_model_params(modelname=modelname, folder=folder)


    return params, postprocess_data(f, data_len=data_len, scaler=scaler, 
                                    wavelet=params["WT_OUTPUT"], **data_processing_params)


def train_test_split(X, y, xt=None, train_perc=0.8, batch_xt=False):
    """
    Splits u, xt and g_u into training set.

    Params:
        @ batch_xt: trunk in batches

    if batch_xt:
        @ u.shape = [bs, x_len]
        @ xt.shape = [bs, x_len*t_len, 3]
        @ g_u.shape = [bs, x_len*t_len] 
    else:
        @ u.shape = [bs, x_len]
        @ xt.shape = [x_len*t_len, 2]
        @ g_u.shape = [bs, x_len*t_len] 
    """
    
    def _split(f, train_size):
        """
        Splits f into train and test sets.
        """
        if isinstance(f, (list, tuple)):
            train, test = list(), list()
            for i, f_i in enumerate(f):
                train.append(f_i[:train_size])
                test.append(f_i[train_size:])
                assert(train[i].shape[-1]==test[i].shape[-1])
        else:            
            train, test = f[:train_size], f[train_size:]
            assert(train.shape[-1]==test.shape[-1])

        return train, test

    if train_perc > 0.0:
        if isinstance(X, (list, tuple)):
            train_size = X[0].shape[0]
        else:
            train_size = X.shape[0]
        train_size = int(np.floor(int(train_size)*train_perc))

        X_train, X_test = _split(X, train_size)
        y_train, y_test = _split(y, train_size)

        if batch_xt:
            xt_train, xt_test = _split(xt, train_size)
        else:
            xt_train, xt_test = xt, xt

        return X_train, y_train, X_test, y_test, xt_train, xt_test
    
    else:
        return None, None, X, y, xt, xt


def shift_g_u(g_u, u=None, x_len=50, t1_as_u0=True, t1_as_zeros=False):
    """
    Shifts g_u of shape [bs, x_len*t_len] by one timestamp.
    Output shape is [bs, x_len*t_len - x_len] or the same as original if t1 is filled in with another value.
    """
    g_u_shifted = g_u[:, x_len:]
    
    if t1_as_u0 and t1_as_zeros:
        warnings.warn("Both 't1_as_u0' and 't1_as_zeros' set to True: Using 't1_as_u0'.", UserWarning)
    
    if t1_as_u0:
        if u is None:
            raise Exception("Found u is None. u must be an array if t1_as_u0 is True.")
            
        g_u_shifted = np.concatenate([u, g_u_shifted], axis=1)
    
    elif t1_as_zeros:
        t1_padding = np.zeros([g_u.shape[0], x_len])
        g_u_shifted = np.concatenate([t1_padding, g_u_shifted], axis=1)
        assert(g_u.shape==g_u_shifted.shape)
        
    return g_u_shifted


"""--------------------   SUBSETTING/RESAMPLING   --------------------"""


def subset_ts(u, xt, g_u, ts):
    """
    Subsets the data to the solutions (u.shape[1] spatial points) for a given timestamp.
    """
    x_len = u.shape[1]  
    i0 = ts * x_len
    i1 = i0 + x_len
    xt_i = xt[:, i0:i1, :]
    g_u_i = g_u[:, i0:i1]
    
    return u, xt_i, g_u_i



def resample_g_xt(g_u, xt, i, except_i=False):
    """
    Resamples g_u and xt to a lower resolution.
    Assumption: u and g_u are at the same x locations.
    
    Params:
        @ i:            i for every ith timestamp to be sampled
        @ except_i:     sample every timestep that is not i (for testing)
    """
    
    # g_u:
    g_u_reshaped = reshape_g_u(g_u, xt)
    if not except_i:
        g_u_i = g_u_reshaped[:, ::i]
    else:
        r = np.array(range(g_u_reshaped.shape[1]))
        g_u_i = np.delete(g_u_reshaped, r[::i], axis=1)
    g_u_sampled = g_u_i.reshape(g_u_i.shape[0], g_u_i.shape[1]*g_u_i.shape[2])
    
    # xt:
    xt_reshaped = np.array(
        np.split(xt, np.unique(xt[:, 1], return_index=True)[1][1:]))
    
    if not except_i:
        xt_i = xt_reshaped[::i, :]
    else:
        r = np.array(range(xt_reshaped.shape[0]))
        xt_i = np.delete(xt_reshaped, r[::i], axis=0)

    xt_sampled = xt_i.reshape((xt_i.shape[0]*xt_i.shape[1], xt_i.shape[2]))

    return g_u_sampled, xt_sampled

def resample_into_testing_sets(g_u, xt, i=3, split_into_two=False):
    g_u_test, xt_test = resample_g_xt(g_u=g_u, xt=xt, i=i, except_i=True)
    
    if not split_into_two:
        return g_u_test, xt_test
    
    else:
        g_u_test1, xt_test1 = resample_g_xt(g_u=g_u_test, xt=xt_test, i=2)
        g_u_test2, xt_test2 = resample_g_xt(g_u=g_u_test, xt=xt_test, i=2, except_i=True)
        
        return [g_u_test1, g_u_test2], [xt_test1, xt_test2]


"""--------------------   TRUNK FUNCTIONS   --------------------"""

def make_xt(x_len, t_len, x_2d=False):
    """
    Makes a 2D trunk input of the form:
    x: (x1, x2...xn, x1, x2...xn...xn)
    t: (t1, t1...t1, t2, t2...t2...tn)
    """
    if x_2d:
        x_len = int(np.sqrt(x_len))

    x = np.array(range(1, x_len+1))
    t = np.array(range(1, t_len+1))

    if x_2d:
        x_col1 = np.tile(np.repeat(x, x.shape[0]), t.shape[0])
        x_col2 = np.tile(np.tile(x, x.shape[0]), t.shape[0])
        t_col = np.repeat(t, x.shape[0]**2)
        xt = np.stack([t_col, x_col1, x_col2]).T
    else:
        x_col = np.tile(x, t.shape[0])
        t_col = np.repeat(t, x.shape[0])
        xt = np.stack([x_col, t_col]).T
    
    return xt




"""--------------------      RESHAPING      --------------------"""


def reshape_to_time_intervals(data, t_len):
    """
    Reshape from [total_n_timesteps, n_features] to [n_instances, t_len, n_features]
    t_len is the selected length of the interval
    """
    n_timesteps = int(np.floor(data.shape[0]/t_len)*t_len)
    data = data[:n_timesteps]
    data = data.reshape(-1, t_len, data.shape[1])
    return data

def reshape_3d_2d(g_u):
    """
    Reshape g_u from [bs, t_len, x_len] to [bs, t_len*x_len]
    If the array is not 3D, do nothing.
    """
    def _reshape(g_u):
        if len(g_u.shape)==3:
            g_u = g_u.reshape(g_u.shape[0], g_u.shape[1]*g_u.shape[2])
        return g_u
    
    if isinstance(g_u, (list, tuple)):
        g_u = [_reshape(g_u_i) for g_u_i in g_u]
    else:
        g_u = _reshape(g_u)

    return g_u

def reshape_g_u2(g_u, x_len=50):
    """
    Reshape g_u according to the input shape.
    """
    
    if len(g_u.shape)==3:
        g_u = reshape_3d_2d(g_u)
    elif len(g_u.shape)==2:
        g_u = reshape_2d_3d(g_u, x_len=x_len)
    else:
        raise ValueError("g_u must be of shape length 2 or 3.")
            
    return g_u


def reshape_2d_3d(g_u, x_len=50):
    """
    Reshape g_u from [bs, t_len*x_len] to [bs, t_len, x_len].
    If the array is not 2D, do nothing.
    """
    if len(g_u.shape)==2:
        g_u = g_u.reshape(g_u.shape[0], int(g_u.shape[1]/x_len), x_len)
    
    return g_u


def reshape_g_u(g_u_output, xt):
    """
    Transforms output functions into an array of shape [bs, t_len, x_len]
    
    params:
        @ g_u_output: DeepONet prediction of shape [bs, t_len * x_len]
        @ xt: trunk input of shape [_,2]
    """

    g_u_output_reshaped = g_u_output.reshape(g_u_output.shape[0],
                                             len(np.unique(xt[:, 1])),
                                             len(np.unique(xt[:, 0])))

    return g_u_output_reshaped


"""--------------------       SCALING       --------------------"""


def scale_data(u, xt, g_u, u_scaler, xt_scaler, g_u_scaler):
    """
    Scaling the data of the format:
        - u: Branch input
        - t: Trunk input
        - g_u: DeepONet output 
    If '*_scaler' is a string: fit and transform; if it is a dictionary, only transform.
    """

    def _multiple_functions(f, scaler):    
        assert(len(f)==len(scaler))      

        f_list, scaler_list = list(), list()
        for f_i, scaler_i in zip(f, scaler):
            f_col, scaler_col = scaling(f_i, scaler_i)
            f_list.append(f_col)
            scaler_list.append(scaler_col)

        return f_list, scaler_list
    
    def _by_column(f, scaler):
        f_list, scaler_list = list(), list()
        for i, scaler_i in enumerate(scaler):
            f_col, scaler_col = scaling(f[:, i], scaler_i)
            f_list.append(f_col)
            scaler_list.append(scaler_col)
        f_scaled = np.stack(f_list).T

        return f_scaled, scaler_list

    
    scaled_functions, scalers = list(), list()

    for i, (f, scaler) in enumerate(zip([u, xt, g_u], [u_scaler, xt_scaler, g_u_scaler])):

        if isinstance(f, (tuple, list)) and (not isinstance(scaler, (tuple, list))):
            scaler = [scaler for i in range(len(f))]

        try:
            if (isinstance(scaler, (dict, str))) or (scaler is None):
                f_scaled, scaler = scaling(f, scaler)

            elif isinstance(scaler, (list, tuple)):
                f_scaled, scaler = _multiple_functions(f, scaler)

            else:
                print("Scaler must be a dictionary or a string, or list of dictionaries and strings.")

        except Exception as err:
            print(f"Error at {i}: {err=}, {type(err)=}")
            raise

        scaled_functions.append(f_scaled)
        scalers.append(scaler)

    u, xt, g_u = scaled_functions
    u_scaler, xt_scaler, g_u_scaler = scalers

    return u, xt, g_u, u_scaler, xt_scaler, g_u_scaler

def scale_indiv_features(features, scaler="standard", axis=2):

    scaler_list = []
    for i in range(features.shape[axis]):
        if axis==2:
            if isinstance(scaler, list):
                features[:,:,i], scaler_feature = scaling(features[:,:,i], scaler[i])
            else:
                features[:,:,i], scaler_feature = scaling(features[:,:,i], scaler)
        elif axis==1:
            if isinstance(scaler, list):
                features[:,i], scaler_feature = scaling(features[:,i], scaler[i])
            else:
                features[:,i], scaler_feature = scaling(features[:,i], scaler)
        scaler_list.append(scaler_feature)
    return features, scaler_list


def scaling(f, scaler):
    """
    Scales f either with standard or minmax scaling.
    @param scaler: str or dict
    """

    scaler_type = scaler if (isinstance(scaler, str) or (scaler is None)) else scaler['scaler']

    if (scaler_type is None) or (scaler_type == "None") or (scaler_type == "none"):
        f_scaled = f
        scaler_type = "None"
        scaler_features = dict({})

    elif scaler_type == "standard":
        f_mean, f_std = (f.mean(), f.std()) if isinstance(
            scaler, str) else (scaler['mean'], scaler['std'])
        f_scaled = standard_scaler(f, f_mean, f_std)
        scaler_features = dict({"mean": f_mean, "std": f_std})

    elif scaler_type == "minmax":
        f_min, f_max = (f.min(), f.max()) if isinstance(
            scaler, str) else (scaler['min'], scaler['max'])
        f_scaled = minmax_scaler(f, f_min, f_max)
        scaler_features = dict({"min": f_min, "max": f_max})

    elif scaler_type == "norm":
        f_norm = np.sqrt(np.mean(f**2, axis=1))
        f_scaled = np.divide(f.T, f_norm).T
        scaler_features = dict({})

    else:
        print("ERROR: Scaler must be either None, \"standard\", \"minmax\" or \"norm\".")

    # Scaler info as a dictionary
    scaler_dict = dict({"scaler": scaler_type})
    scaler_dict.update(scaler_features)

    return f_scaled, scaler_dict


def inverse_scaling(z, scaler):

    def _inverse_scaling(z, scaler):
        if scaler['scaler'] == "standard":
            x = standard_scaler_inverse(z, scaler['mean'], scaler['std'])

        elif scaler['scaler'] == "minmax":
            x = minmax_scaler_inverse(z, scaler['min'], scaler['max'])

        elif scaler['scaler'] == "None":
            x = z

        return x

    if isinstance(z, (list, tuple)):
        # TODO: reformulate the condition: doesn't work if the output is a full np array / tensor
        x = [_inverse_scaling(z_i, scaler_i) for z_i, scaler_i in zip(z, scaler)]
    else:
        x = _inverse_scaling(z, scaler)

    return x

def standard_scaler(x, mean, std):
    """
    Not using StandardScaler because it treats each timestamp as a separate feature.
    """
    z = (x - mean) / std
    return z


def standard_scaler_inverse(z, mean, std):
    x = (z * std) + mean

    return x


def minmax_scaler(x, minimum, maximum):
    z = (x - minimum) / (maximum - minimum)

    return z


def minmax_scaler_inverse(z, minimum, maximum):
    x = (z * (maximum - minimum)) + minimum

    return x

def scaling_coef(u, target_integ=1, axis=1, bias=0.2):
    """
    Calculate the coefficient to scale the input such that the integral is equal to target_integ.
    """
    u = u + bias
    int_u = np.sum(u, axis=axis)         # Integral over every sample: shape (bs,)
    coef = (target_integ/int_u)                 # Coefficient to scale input: shape (bs,)
    return coef

def scale_by_coef(y, coef, bias=0.2):
    y = y + bias
    y_shape = y.shape
    coef = np.reshape(np.repeat(coef, y_shape[1]), (-1, y_shape[1]))
    y = y * coef

    return y

def inverse_scale_by_coef(y, coef, bias=0.2):
    y_shape = y.shape
    coef = np.reshape(np.repeat(coef, y_shape[1]), (-1, y_shape[1]))
    y = y / coef
    y = y - bias

    return y



