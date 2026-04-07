from Processor import  POWDERRF_Processor
from Modeling import Conv1D_RF_Classifier
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
import torch
from safetensors.torch import load_file
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from time import perf_counter
import os
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def trainer(model = None, xtrain = np.array([]), ytrain = np.array([]), batch_sz = 100, epochs = 100, lr = 0.001 , savepath = "./", i_checkpoint = 25, device = "cuda", val_split = 0.05):
    """
    tensorboard --logdir=<your_savepath>/tensorboard_logs
    """
    if savepath[-1] != "/": savepath += "/"
    writer = SummaryWriter(log_dir=savepath + "tensorboard_logs")
    
    if val_split != 0:
        val_index = int(np.floor(xtrain.shape[0]*val_split))
        xval = torch.tensor(xtrain[:val_index]).type(torch.float)
        yval = ytrain[:val_index]

        xtrain = xtrain[val_index:]
        ytrain = ytrain[val_index:]

        yval_batched = torch.tensor(yval).type(torch.float)
        xval_batched = torch.split(xval, batch_sz)
        yval_batched = torch.split(yval_batched, batch_sz)
    
        
        
    xtrain_batched = torch.tensor(xtrain).type(torch.float)
    ytrain_batched = torch.tensor(ytrain).type(torch.float)
    xtrain_batched = torch.split(xtrain_batched, batch_sz)
    ytrain_batched = torch.split(ytrain_batched, batch_sz)


    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    model = model.to(device)
    loss_sum = 0
    i = 0
    #time = 0
    history = {}
    for epoch in range(epochs):
        history[epoch] = {}
        print("Epoch:", epoch)
        #t0 = perf_counter()
        for x_batch, y_batch in tqdm(zip(xtrain_batched, ytrain_batched)):
            i += 1
            # Zero your gradients for every batch
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)#y_batch.flatten().to(device)
            #logits should be shape:(batch_sz, hidden_length)
            logits, loss = model(data_in = x_batch, y_true = y_batch)
            # Compute the loss and its gradients
            loss.backward()
            loss_sum += loss.item()
            avg_loss = loss_sum/i

            writer.add_scalar("Loss/batch", loss.item(), i)       
            writer.add_scalar("Loss/avg", avg_loss, i)            
        
            if i%i_checkpoint == 0:
                model_savepath = savepath + "checkpoint-" + str(i) + ".pth"
                torch.save(model.state_dict(), model_savepath)
            # Adjust weights
            optimizer.step()

        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
        print(f"Loss/epoch_avg:{avg_loss}")
        #t1 = perf_counter()
        if val_split != 0:
            model.eval()
            with torch.no_grad():
                preds = []
                pred_probs = []
                for batch in xval_batched:
                    batch = batch.to(device)
                    logits, probs, pred = model(batch)
                    pred_probs.append(probs.detach().cpu().numpy())
                    preds.append(pred.detach().cpu().numpy())
                preds = np.concatenate(preds, axis = 0)
                pred_probs = np.concatenate(pred_probs, axis = 0)
                acc_score = accuracy_score(preds.flatten(), yval.flatten())
                roc_auc = roc_auc_score(yval, pred_probs, multi_class='ovr')
                writer.add_scalar("Metrics/roc_auc", roc_auc, epoch)
                writer.add_scalar("Metrics/acc", acc_score, epoch)
                print(f"Metrics/roc_auc:{roc_auc}")
                print(f"Metrics/acc:{acc_score}")
            model.train()

    model_savepath = savepath + "final_model"+ ".pth"
    torch.save(model.state_dict(), model_savepath)

    return model
    
def evaluator(model = None, xtest = np.array([]), ytest = np.array([]), batch_sz = 32,savepath = "./" ,device = "cuda"):
    model = model.to(device)
    xtest_batched = torch.tensor(xtest).type(torch.float)
    xtest_batched = torch.split(xtest_batched, batch_sz)
    preds = []
    pred_probs = []
    for batch in tqdm(xtest_batched):
        batch = batch.to(device)
        logits, probs, pred = model(batch)
        pred_probs.append(probs.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds, axis = 0)
    pred_probs = np.concatenate(pred_probs, axis = 0)
    report = classification_report(ytest, preds, output_dict = True, target_names = range(len(ytest[0])))
    fpr_micro, tpr_micro, _ = roc_curve(ytest.flatten(), pred_probs.flatten())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    acc = accuracy_score(ytest.flatten(), preds.flatten())
    

    print(f"Metrics/acc:{acc}")
    print(report)
    plt.plot(fpr_micro, tpr_micro)
    plt.text(
            0.30, 0.95,          # (x, y) position in "axes fraction" coordinates
            f"AUROC = {roc_auc_micro:.4f}",  
            transform=plt.gca().transAxes,  # interpret coordinates relative to axes (0–1)
            ha='right', va='top',           # align text properly
            fontsize=11, color='blue'
        )
    plt.show()
    results_path  = savepath + "results.json"

    with open(results_path, 'w') as f:
        json.dump(report, f);
    results = {}
    results["acc"] = acc
    results["fpr_micro"] = fpr_micro
    results["tpr_micro"] = tpr_micro
    results["roc_auc"] = roc_auc_micro
    results["report"] = report

    return results 

def load_tensorboard_logs(log_dir):
    """
    Loads all scalar data from a TensorBoard log directory.
    Returns a dict of DataFrames, one per scalar tag.
    Usage:
    logs = load_tensorboard_logs("<your_savepath>/tensorboard_logs")

    # Access individual metrics
    batch_loss   = logs["Loss/batch"]
    avg_loss     = logs["Loss/avg"]
    epoch_loss   = logs["Loss/epoch_avg"]
    roc_auc      = logs["Metrics/roc_auc"]

    """
    ea = EventAccumulator(log_dir)
    ea.Reload()  # loads all data from disk

    available_tags = ea.Tags()["scalars"]
    print("Available tags:", available_tags)

    all_data = {}
    for tag in available_tags:
        events = ea.Scalars(tag)
        df = pd.DataFrame(events)  # columns: wall_time, step, value
        all_data[tag] = df

    return all_data

"""
def main(args):
    
    return 0;

if __name__ == "__main__":
    args = Parse_Args();
    main(args);
"""
    