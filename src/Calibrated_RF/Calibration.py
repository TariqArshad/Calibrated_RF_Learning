#import torch
import numpy as np
import matplotlib.pyplot as plt
import torch


def FNR_Calibrator(model, x_calib, y_calib, device = "cuda", alpha = [0.10], lambda_step_sz = 0.05, batch_sz = 32, plot = True):
    """
    Calibration function for classifiers models using conformal risk control
    """
    model.eval()
    x_calib_batched = torch.tensor(x_calib).type(torch.float)
    x_calib_batched = torch.split(x_calib_batched, batch_sz)
    pred_probs = []
    for batch in x_calib_batched:
        logits, probs, preds = model(batch.to(device))
        pred_probs.append(probs.detach().cpu().numpy())
    pred_probs = np.concatenate(pred_probs, axis = 0)
    
    FNR_dict = {}
    FNR_risk_list = []
    lambda_steps = np.arange(0, 1 + lambda_step_sz, lambda_step_sz)
    for i in lambda_steps[:-1]:
        pred_sets = np.uint16(pred_probs > (1.0 - i))
        ytrue_pred_intersect = pred_sets&y_calib
        FNR_loss = 1 - (np.sum(ytrue_pred_intersect, axis = -1)/np.sum(y_calib, axis = -1))
        FNR_risk = np.mean(FNR_loss).item()
        # FNR dict keys are conf levels
        FNR_dict[1.0 - i] = {"loss_dist": FNR_loss, "set_size":np.sum(pred_sets, axis = 1) ,"risk":FNR_risk}
        FNR_risk_list.append(FNR_risk)
        #appending for last confidence level of 0 where model should return all classes making FNR risk 0
    FNR_risk_list.append(0)
    FNR_dict["calib"] = []
    for a in alpha:
        lambda_thresh = lambda_steps[np.array(FNR_risk_list) < a][0]
        FNR_dict["calib"].append((a, lambda_thresh))

    if plot:
        # Get a colormap with as many colors as you need
        colors = plt.cm.viridis(np.linspace(0, 1, len(alpha)))
        plt.plot(lambda_steps, FNR_risk_list)
        plt.title("FNR Risk Calibration Curve")
        for i in range(len(alpha)):
            alpha_i = FNR_dict["calib"][i][0]
            lambda_i = FNR_dict["calib"][i][1]
            plt.axhline(alpha_i, linestyle='--', linewidth=2, label = fr'$\alpha_{{{i}}} = {alpha_i}$', color = colors[i]) 
            plt.axvline(lambda_i, linestyle='--', linewidth=2, label =  fr'$\lambda_{{{i}}} = {lambda_i}$', color = colors[i]) 
        plt.xlabel("\u03BB(1 - confidence)")
        plt.ylabel("Risk")
        plt.legend()
        plt.show()
    return FNR_dict

def Anomaly_Calibrator(model, x_calib, y_calib, device = "cuda", alpha = [0.10], batch_sz = 32, plot = True):
    """
    Calibration function for anaomaly detection models using conformal prediction
    """

    anomaly_dict = {"calib":[]}
    model.eval()
    x_calib_batched = torch.tensor(x_calib).type(torch.float)
    x_calib_batched = torch.split(x_calib_batched, batch_sz)
    pred_probs = []
    for batch in x_calib_batched:
        logits, probs, preds = model(batch.to(device))
        pred_probs.append(probs.detach().cpu().numpy())
    pred_probs = np.concatenate(pred_probs, axis = 0)
    # adjust so max is 0.99 instead of 1.0
    pred_probs = pred_probs*0.99
    bernoulli_entropy = -pred_probs*np.log(pred_probs) - (1-pred_probs)*np.log(1-pred_probs)
    total_uncertainty = np.sum(bernoulli_entropy, axis = -1)
    anomaly_dict["entropy"] = total_uncertainty
    for a in alpha:
        q_level =  np.ceil((total_uncertainty.shape[0]+1)*(1-a))/total_uncertainty.shape[0]
        qhat = np.quantile(total_uncertainty, q_level, axis = 0, method = "higher")
        anomaly_dict["calib"].append((a, qhat))
    if plot:
        # Get a colormap with as many colors as you need
        colors = plt.cm.viridis(np.linspace(0, 1, len(alpha)))
        plt.hist(total_uncertainty, bins = 50)
        i = 0
        for a, qhat in anomaly_dict["calib"]:
            plt.axvline(x= qhat, ls = "--", label = fr'$\alpha$ = {a}, $\hat{{q}}$ = {qhat:.2f}', color = colors[i])
            i += 1
        plt.legend()
        plt.show()

    return anomaly_dict
