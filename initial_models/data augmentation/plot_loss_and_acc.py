import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import pandas as pd
import torch
import torchmetrics

def plot_loss_and_acc(
    log_dir, loss_ylim=(0.0, 1.0), acc_ylim=(0.0, 1.0), recall_ylim=(0.0, 1.0), precision_ylim=(0.0, 1.0), f1score_ylim=(0.0, 1.0),  save_loss=None, save_acc=None
):

    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )
    
    min_loss = df_metrics["val_loss"].min()
    opt_epoch = df_metrics.index[df_metrics['val_loss']==min_loss].values
    plt.axvline(x = opt_epoch, color = 'g', ls='--')
    plt.text(opt_epoch, min_loss, f'  Min. loss {round(min_loss,2)}' f'\n  at epoch {int(opt_epoch)}')


    plt.ylim(loss_ylim)
    if save_loss is not None:
        plt.savefig(save_loss)

    df_metrics[["train_accuracy", "val_accuracy"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.ylim(acc_ylim)
    if save_acc is not None:
        plt.savefig(save_acc)
        
        
        
def confusion_matrix(all_predicted_labels, all_true_labels):
    
    confmat = torchmetrics.ConfusionMatrix(task = 'binary')
    cm = confmat(all_predicted_labels, all_true_labels)
    
    tn = cm[0,0].numpy()
    fp = cm[0,1].numpy()
    fn = cm[1,0].numpy()
    tp = cm[1,1].numpy()

    accuracy = (tn + tp) / (tn + fp + tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * (precision * recall) / (precision + recall)
    
    fig, ax = plot_confusion_matrix(conf_mat=cm.numpy(),
                                   hide_spines=False,
                                   hide_ticks=False,
                                   figsize=None,
                                   cmap=None,
                                   colorbar=False,
                                   show_absolute=True,
                                   show_normed=False,
                                   norm_colormap=None,
                                   class_names=None,
                                   figure=None,
                                   axis=None,
                                   fontcolor_threshold=0.5,)
    plt.show()
    
    print('true negatives (TN):', tn, end='\n')
    print('false positives (FP):', fp, end='\n')
    print('false negatives (FN):', fn, end='\n')
    print('true positives (TP):', tp, end='\n')

    print('accuracy:', accuracy, end='\n')
    print('precision:', precision, end='\n')
    print('recall:', recall, end='\n')
    print('f1score:', f1score, end='\n')
    
    
    
def parcel_level(test_csv, all_predicted_labels):
    test_df_1 = pd.read_csv(test_csv, usecols = ["label", "parcel_id"])
    test_df_2 = test_df_1
    
    # Predictions parcel level df
    
    ## Only retain ground truth at parcel level
    test_df_1 = test_df_1.drop_duplicates(subset=['parcel_id'])
    
    ## Add y_hat to a pandas df
    y_hat = all_predicted_labels.numpy()
    test_df_2["y_hat"] = y_hat
    test_df_2 = test_df_2[["y_hat", "parcel_id"]]
    
    test_df_2 = test_df_2.groupby(['parcel_id'])['y_hat'].agg(pd.Series.mode)
    test_df_2 = test_df_2.reset_index()
    
    test_df_2 = test_df_2.explode(column = 'y_hat')
    
    test_df_2 = test_df_2.sort_values('y_hat').groupby('parcel_id').tail(1)
    test_df_2 = test_df_2.sort_values('parcel_id')
    test_df_2['y_hat'] = test_df_2['y_hat'].astype('float')
    
    # Ground truth parcel level df
    
    test_df_1 = test_df_1.reset_index()
    test_df_1 = test_df_1[['label','parcel_id']]
    test_df_1 = test_df_1.sort_values('parcel_id')
    test_df_1['label'] = test_df_1['label'].astype('float')
    
    #y_hat = torch(test_df_2.y_hat)
    y_hat = torch.tensor(test_df_2['y_hat'].values)
    y = torch.tensor(test_df_1['label'].values)
    
    acc = torch.mean((y_hat == y).float())
    print(f'Parcel level accuracy: {acc:.4f} ({acc*100:.2f}%)')
    
    confusion_matrix(y_hat, y)
    
def parcel_level_new(test_csv, all_predicted_labels):
    test_df_1 = pd.read_csv(test_csv, usecols = ["label", "parcel_id"])
    test_df_2 = test_df_1
    
    # Predictions parcel level df
    
    ## Only retain ground truth at parcel level
    test_df_1 = test_df_1.drop_duplicates(subset=['parcel_id'])
    
    ## Add y_hat to a pandas df
    y_hat = all_predicted_labels.numpy()
    test_df_2["y_hat"] = y_hat
    test_df_2 = test_df_2[["y_hat", "parcel_id"]]
    
    test_df_2 = test_df_2.groupby('parcel_id').agg(lambda parcel_id: parcel_id.mode().max())
    test_df_2 = test_df_2.reset_index()
    test_df_2 = test_df_2.sort_values('parcel_id')
    test_df_2['y_hat'] = test_df_2['y_hat'].astype('float')
    
    # Ground truth parcel level df
    
    test_df_1 = test_df_1.reset_index()
    test_df_1 = test_df_1[['label','parcel_id']]
    test_df_1 = test_df_1.sort_values('parcel_id')
    test_df_1['label'] = test_df_1['label'].astype('float')
    
    #y_hat = torch(test_df_2.y_hat)
    y_hat = torch.tensor(test_df_2['y_hat'].values)
    y = torch.tensor(test_df_1['label'].values)
    
    acc = torch.mean((y_hat == y).float())
    print(f'Parcel level accuracy: {acc:.4f} ({acc*100:.2f}%)')
    
    confusion_matrix(y_hat, y)