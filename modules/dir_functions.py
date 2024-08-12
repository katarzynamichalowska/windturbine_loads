import os
import numpy as np
import pandas as pd
from datetime import datetime


def make_output_dir(params, test_dir="../models/TEST", is_testing=False):
    """
    Makes an output folder for the model.
    """
    
    if is_testing:
        output_folder = test_dir
    else:
        output_folder = params.OUTPUT_FOLDER + "_" + timestamp_now()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    elif is_testing:
        os.makedirs(output_folder, exist_ok=True)
    else:
        if not params.START_FROM_LATEST:
            output_folder = output_folder + "_copy"
            os.makedirs(output_folder, exist_ok=True)

    return output_folder

def timestamp_now():
    """
    Produces a timestamp of today's date and time.
    """
    timestamp = str(datetime.now())[2:22].replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    timestamp_now = timestamp[:6] + "_" + timestamp[6:]

    return timestamp_now

def timestamp_today():
    """
    Produces a timestamp of today's date.
    """
    timestamp = str(datetime.now())[2:22].replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    timestamp_date = timestamp[:6]

    return timestamp_date

def models_today(modelname="deeponet", directory="../models"):
    """
    Returns a list of model names that contain the modelname and today's date.
    """
    
    model_names = models_date(modelname=modelname, date=timestamp_today(), directory=directory)
    
    return model_names

def models_date(modelname="deeponet", date="221005", directory="../models"):
    """
    Returns a list of model names that contain the modelname and the date.
    """
    model_names = [d for d in os.listdir(directory) if (".DS" not in d) and (".ipynb" not in d)]
    model_names = [m for m in model_names if f"{modelname}_{date}" in m]
    model_names.sort()
    
    return model_names

def make_paths_in_dir(folder, files):
    """
    Params:
        @files: list of files to join paths
    """
    file_paths = list()
    for f in files:
        file_paths.append(os.path.join(folder, f))

    return file_paths


def correct_dtype(val):
    """
    Corrects the data type of the value.
    """

    if val is not None:
        if is_float(val):
        # Float
            val = float(val)
            if val-int(val)==0:
            # Int
                val = int(val)
        elif val=="True":
        # Bool
            val = True
        elif val=="False":
        # Bool
            val = False
        elif isinstance(val, str):
        # String
            if val=="None":
                val = None
            else:
                val = val.strip("\"")
                if ("[" in val) and ("]" in val):
                    val = val.replace("[", "").replace("]", "").replace("\'", "")
                    if "," in val:
                        val = val.split(", ")
                    else:
                        val = [val]

    return val

def read_params_module_as_dict(module_name):
    """
    Reads all parameters from a module as a dictionary.
    """
    module = globals().get(module_name, None)
    if module:
        parameters = {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
    return parameters

def summarize_iterate(models_folder, func, filename):
    """
    Summarize parameters or metrics of existing models into a dataframe.

    Params:
        @ models_folder: folder to read the models from
        @ func: function to use for summary: e.g., read_parameters() or read_metrics_log().
        @ filename: how the log/params file is named in the "models_folder/model_name_timestamp" folder.
    """
    model_names = os.listdir(models_folder)

    params_list = list()
    for i, model in enumerate(model_names):
        params_path = os.path.join(os.path.join(models_folder, model), filename)
        if os.path.isfile(params_path):
            params = func(params_path)
            params = pd.DataFrame(params, index=[i])
            params.columns = params.columns.str.lower()
            params["model"] = model
            params_list.append(params)

    params_df = pd.concat(params_list, axis=0)

    return params_df


def summarize_params(models_folder):
    """
    Summarize the parameters of each model in models_folder into a dataframe.
    """

    return summarize_iterate(models_folder, read_parameters, "params.txt")


def summarize_metrics(models_folder):
    """
    Summarize the metrics of each model in models_folder into a dataframe.
    """
    try:
        metrics = summarize_iterate(models_folder, read_metrics_log, "log.out")
    except:
        try:
            metrics = summarize_iterate(models_folder, read_metrics_log, "logs.out")
        except:
            print("Exception")
            
    return metrics


def read_parameters(params_path):
    """
    Reads parameters to a dictionary from a params.txt file.
    """
    d = {}
    with open(params_path) as f:
        for line in f:
            keyval = line.split(" = ")
            key = keyval[0]
            val = keyval[1:]
            val = " = ".join(val)
            val = val.strip("\n")
            val = correct_dtype(val)
            d[key] = val
    return d

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def parameter_from_logs(modelname, param, folder="../models", logfile="log.out"):
    """
    Read a parameter from the logs of a model.
    """
    param_path = os.path.join(os.path.join(folder, modelname), logfile)

    with open(param_path,"r") as fi:
        for ln in fi:
            if ln.startswith(param):
                param_value = ln[len(param):].split("\t")[1].split("\n")[0].strip(" ")
                param_value = correct_dtype(param_value)
    return param_value
       

def read_model_params(params_to_read, modelname, folder="../models", logfile="log.out"):
    """
    Read parameters from the logs of a model.
    """
    params_vals= list()

    for p in params_to_read:
        try:
            params_vals.append(parameter_from_logs(modelname=modelname, param=p.upper(), folder=folder, logfile=logfile))
        except:
            try:
                params_vals.append(parameter_from_logs(modelname=modelname, param=p.lower(), folder=folder, logfile=logfile))
            except:
                try:
                    params_vals.append(getattr(params_default, p.upper()))
                except:
                    print(f"Couldn't find {p} for model {modelname}.")

    return params_vals

def read_metrics_log(log_path):
    """
    Read testing results (metrics r2, rse, rmse, mae) from the logs.
    """
    textfile = open(log_path)
    lines = textfile.readlines()
    metrics_dict = dict()
    metrics = ["R2", "RSE", "RMSE", "MAE"]
    for i, ln in enumerate(reversed(lines)):
        for m in metrics:
            if ln.startswith(m):
                values = ln.strip("\n").replace("\t", " ").split(" ")
                if values[1]=="train:":
                    metrics_dict.update({f"{m}_train": float(values[2].strip(","))})
                if values[3]=="test:":
                    metrics_dict.update({f"{m}_test": float(values[4].strip(","))})

        if ("TESTING" in ln) or (i>15):
            break

    return metrics_dict

def read_checkpoints(checkpoint_dir):
    """
    Read the checkpoint numbers from the checkpoint directory.
    Returns:
        @ cp_numbers: list of checkpoint numbers (int)
    """
    cp_files = os.listdir(checkpoint_dir)
    cp_int = np.array([i.split("-")[1].split(".")[0] for i in cp_files if i!="checkpoint"])
    cp_numbers = np.unique(cp_int.astype('int32'))

    return cp_numbers

def read_max_checkpoint(checkpoint_dir):
    """
    Read the maximum checkpoint number from the checkpoint directory.
    Returns:
        @ cp_max: maximum checkpoint number (int)
    """
    
    return read_checkpoints(checkpoint_dir)[-1]