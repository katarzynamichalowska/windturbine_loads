import io
import os
from modules.data_manipulation import inverse_scaling
from modules.testing import compute_metrics
import numpy as np
import datetime
import pprint

def add_loss_info(g_loss, loss_name, loss_value, lambda_value):
    if lambda_value > 0:
        fool_loss = g_loss.item() - (lambda_value * loss_value.item())
        return f"\t[G fool loss: {fool_loss:.4f}]\t[G {loss_name}: {loss_value.item():.4f}]"
    return ""


def print_time(text="Time"):
    string = "{}: {}\n".format(text, datetime.datetime.now())

    return string


def pprint_layer_dict(layers):
    layers = pprint.pformat(layers, indent=1, compact=False, sort_dicts=False)
    layers = '\t'+'\t'.join(layers.splitlines(True))
    return layers


def print_info(params):

    string = "\n\n#--------------------       INFO       --------------------#\n\n"

    branch_hidden_layers = pprint_layer_dict(params.BRANCH_HIDDEN_LAYERS)
    trunk_hidden_layers = pprint_layer_dict(params.TRUNK_HIDDEN_LAYERS)
    if params.RNN_LAYERS is not None:
        rnn_layers = pprint_layer_dict(params.RNN_LAYERS)

    string += "BRANCH: \n{}\n\n".format(branch_hidden_layers)
    string += "TRUNK: \n{}\n\n".format(trunk_hidden_layers)
    if params.RNN_LAYERS is not None:
        string += "RNN: \n{}\n\n".format(rnn_layers)
    string += "B-T OUTPUT: \n\t{}\n\n".format(params.BT_OUTPUT_SIZE)
    string += "SCALING: \n\tu:\t\t{}\n\txt:\t\t{}\n\tg_u:\t{}\n\n".format(params.U_SCALER,
                                                                          params.XT_SCALER,
                                                                          params.G_U_SCALER)
    string += "TRAINING: \n\tlearning rate:\t{}\n\tn_epochs:\t\t{}\n\tbatch_size:\t\t{}\n\n".format(params.LEARNING_RATE,
                                                                                                    params.N_EPOCHS,
                                                                                                    params.BATCH_SIZE)
    string += "NOTES: \n\t{}".format(params.NOTES)

    print(string)

    return string


def print_params(module_name, table=False, header=False):
    items = dict([(item, getattr(module_name, item))
                 for item in dir(module_name) if not item.startswith("__")])
    string = ""
    if header:
        string += "\n\n#--------------------     PARAMETERS     --------------------#\n\n"
    if table:
        string += table_dictionary_items(items)
    else:
        string += equal_dictionary_items(items)

    print(string)

    return string


def table_dictionary_items(items):
    string = ""
    for key in items:
        string += "{:<30}:\t{:<50}\n".format(str(key), str(items[key]))
    return string


def equal_dictionary_items(items):
    string = ""
    for key in items:
        value = items[key]
        value = f"\"{value}\"" if isinstance(value, str) else value
        string += "{} = {}\n".format(str(key), str(value))
    return string


def print_model_summary(model, modelname):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    string = stream.getvalue()
    stream.close()
    header = "\n\n#--------------------        {}        --------------------#\n\n".format(
        modelname.upper())

    string = header + string
    print(string)

    return string

def print_activations(model, modelname):

    header = "\n\n#--------------------        {}        --------------------#\n\n".format(
        modelname.upper() + " ACTIVATIONS")

    string = header

    for i, layer in enumerate (model.layers):
        layer_name = str(i) + "\t" + str(layer).split(".")[-1].split(" ")[0]
        try:
            activation = str(layer.activation).split(" ")[1]
            string += "{:<20}:\t{:<20}\n".format(str(layer_name), str(activation))
        except AttributeError:
            string +=  "{:<20}:\t{:<20}\n".format(str(layer_name), "---")#f"{layer_name}\n"

    return string


def print_scale_info(u_train, u_test, g_u_train, g_u_test, xt_train, xt_test):

    def _min_range(variable):
        min_value = np.round(variable.min(axis=0).min(), 5)
        max_value = np.round(variable.min(axis=0).max(), 5)
        
        return min_value, max_value

    def _max_range(variable):
        min_value = np.round(variable.max(axis=0).min(), 5)
        max_value = np.round(variable.max(axis=0).max(), 5)

        return min_value, max_value
    
    def _string_ranges(name, variable):
        string  = name + ":\n"
        string += "\tmin: ({})\n".format(_min_range(variable))
        string += "\tmax: ({})\n".format(_max_range(variable))

        return string

    def _print_ranges(name, variable):
        string = ""
        if isinstance(variable, (list, tuple)):
            for i, v in enumerate(variable):
                n = f"{name}_{i}"
                string += _string_ranges(n, v)
        else:
             string += _string_ranges(name, variable)
        
        return string

    string = "\n\n#--------------------      SCALING      --------------------#\n\n"

    string += "Training data scale (min-max values range per column):\n\n"
    for name, variable in zip(["u", "g_u", "xt"], [u_train, g_u_train, xt_train]):
        string += _print_ranges(name, variable)

    string += "\n\nTesting data scale (min-max values range per column):\n\n"
    for name, variable in zip(["u", "g_u", "xt"], [u_test, g_u_test, xt_test]):
        string += _print_ranges(name, variable)

    return string


def print_training_history(history):
    string = "\n\n#--------------------      TRAINING      --------------------#\n\n"
    string += history

    return string


def print_testing(g_u_train, g_u_train_pred, g_u_test, g_u_test_pred, g_u_scaler=None, header=True, text=""):
    if header:
        string = "\n\n#--------------------     TESTING     --------------------#\n\n"
    else:
        string = ""

    compute_train = (g_u_train is not None) and (g_u_train_pred is not None)
    compute_test = (g_u_test is not None) and (g_u_test_pred is not None)

    if g_u_scaler is not None:
        if compute_train:
            g_u_train = inverse_scaling(g_u_train, g_u_scaler)
            g_u_train_pred = inverse_scaling(g_u_train_pred, g_u_scaler)
        if compute_test:
            g_u_test = inverse_scaling(g_u_test, g_u_scaler)
            g_u_test_pred = inverse_scaling(g_u_test_pred, g_u_scaler)

    if compute_train:
        metrics_train = compute_metrics(g_u_train, g_u_train_pred)
    if compute_test:
        metrics_test = compute_metrics(g_u_test, g_u_test_pred)

    if compute_train and compute_test:
        string += "MAE train: {},\ttest: {}\n".format(
            metrics_train['mae'], metrics_test['mae'])
        string += "RMSE train: {},\ttest: {}\n".format(
            metrics_train['rmse'], metrics_test['rmse'])
        string += "RSE train: {},\ttest: {}\n".format(
            metrics_train['rse'], metrics_test['rse'])
        #string += "R2 train: {},\ttest: {}\n".format(
        #    metrics_train['r2'], metrics_test['r2'])
        string += "\n\n"
    
    elif compute_test:
        string += f"Test {text}:\n"
        string += "MAE:\t{}\t".format(np.round(metrics_test['mae'],3))
        string += "RMSE:\t{}\t".format(np.round(metrics_test['rmse'],3))
        string += "RSE:\t{}\t".format(np.round(metrics_test['rse'],3))
        #string += "R2:\t{}\t".format(np.round(metrics_test['r2'],3))
        string += "\n"

    elif compute_train:
        string += f"Train {text}:\n"
        string += "MAE:\t{}\t".format(np.round(metrics_train['mae'],3))
        string += "RMSE:\t{}\t".format(np.round(metrics_train['rmse'],3))
        string += "RSE:\t{}\t".format(np.round(metrics_train['rse'],3))
        #string += "R2:\t{}\t".format(np.round(metrics_train['r2'],3))
        string += "\n"

    print(string)

    return string


def parameter_from_logs(folder, param):
    with open(os.path.join(folder, "log.out"), "r") as fi:
        for ln in fi:
            if ln.startswith(param):
                param_value = ln[len(param):].split(
                    "\t")[1].split("\n")[0].strip(" ")

    return param_value
