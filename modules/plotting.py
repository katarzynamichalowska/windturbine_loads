import matplotlib.pyplot as plt
import numpy as np
import os


def plot_gan_samples(real_data, generated_data, x=None, num_pairs=8, figsize=(6, 5), plot_name="sample_plot", 
                     output_folder=".", titles_list=None, log=False):
    """
    Plot real and generated samples on the same plot with different colors.
    
    Parameters:
    - real_data: Tensor of real samples.
    - generated_data: Tensor of generated samples.
    - num_pairs: Number of samples to display.
    - figsize: Size of the figure.
    """
    #if not real_data.is_cpu:
    #    real_samples = real_data.detach().cpu()
    #if not generated_data.is_cpu:
    #    generated_samples = generated_data.detach().cpu()
    
    plt.figure(figsize=figsize)
    for i in range(num_pairs):
        # Plot real and generated data on the same subplot
        plt.subplot(num_pairs, 1, i + 1)
        if x is None:
            plt.plot(real_data[i], label='Real', color='blue')
            plt.plot(generated_data[i], label='Generated', color='red')
        else:
            plt.plot(x, real_data[i], label='Real', color='blue')
            plt.plot(x, generated_data[i], label='Generated', color='red')
        plt.axis('off')
        if titles_list is not None:
            plt.title(titles_list[i], fontsize=8)
        if i == 0:
            plt.legend(loc='upper right')
        if log:
            plt.yscale('log')
            plt.xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{plot_name}.pdf'))
    plt.close()


def color_scatter(x_label, y_label, color, info, save_path, title=""):
    """
    Scatter plot of the given x and y with color c.
    """
    x = info[x_label]
    y = info[y_label]
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=color, cmap='viridis', alpha=0.7)
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def info_scatter(x, y, x_label, y_label, save_path, title=""):
    """
    Scatter plot of the given x and y with color c.
    """
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label=y_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def subplots_prediction(y, y_pred=None, idx_list=[0,1,2,3,4,5], nr_subplots=6, save_path=".", main_title=None, titles=None,
                        legend=True):
    """
    Plot the true and predicted values for the given indices.
    """
    nrows = int(np.floor(np.sqrt(nr_subplots)))
    ncols = int(nr_subplots/nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(int(ncols*3), int(nrows*3)), sharex=True, sharey=True)
    handles, labels = [], []

    for i, (idx, ax) in enumerate(zip(idx_list, axes.flatten())):
        if idx < y.shape[0]:
            for s in range(y.shape[2]):
                line, = ax.plot(y[idx, :, s], label="True_{}".format(s), c="red")
            if i==0:
                handles.append(line)
                labels.append("True")
            if y_pred is not None:
                for p in range(y_pred.shape[2]):
                    line, = ax.plot(y_pred[idx, :, p], label="Predicted_{}".format(p), c="black", linestyle="--")
                if i==0:
                    handles.append(line)
                    labels.append("Predicted")

            if idx < ncols*(nrows-1):
                ax.set_xticks([])
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    
    if main_title is not None:
        fig.suptitle(main_title, fontsize=10)
    if legend:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=False, ncol=2)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_history(history, logx=False, logy=False, savedir=None, plot_val=True, plot_train=True,
                 variance=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if plot_train:
        if variance is None:
            ax.plot(history['train_loss'], label="Training loss")
        else:
            ax.plot(history['train_loss']/variance, label="Training loss")
    if plot_val:
        if variance is None:
            ax.plot(history['val_loss'], label="Validation loss")
        else:
            ax.plot(history['val_loss']/variance, label="Validation loss")
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.legend()
    if variance is None:
        ax.set_ylabel("MSE")
    else:
        ax.set_ylabel("RSE")
    if savedir is not None:
        plt.savefig(savedir)
    else:
        plt.show()

def plot_history_all(history, output_folder, plot_name, variance=None):
    plot_history(history, logy=False, savedir=os.path.join(output_folder, plot_name+".png"), variance=variance)
    plot_history(history, logy=True, savedir=os.path.join(output_folder, plot_name+"_log.png"), variance=variance)
    plot_history(history, logy=False, plot_train=False, savedir=os.path.join(output_folder, plot_name+"_val.png"), variance=variance)
    plot_history(history, logy=True, plot_train=False, savedir=os.path.join(output_folder, plot_name+"_val_log.png"), variance=variance)

