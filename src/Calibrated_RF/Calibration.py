import numpy as np
import matplotlib.pyplot as plt
import torch


def FNR_Calibrator(
    model,
    x_calib,
    y_calib,
    device="cuda",
    alpha=[0.10],
    lambda_step_sz=0.05,
    batch_sz=32,
    plot=True,
):
    """Calibrate a classifier with conformal risk control on false-negative rate.

    Prediction sets are formed by including every class whose predicted
    probability exceeds ``1 - lambda``. The smallest ``lambda`` whose empirical
    FNR risk stays below each requested ``alpha`` is recorded.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier in eval mode. ``model(x)`` must return
        ``(logits, probs, preds)``.
    x_calib : array-like
        Calibration features with the same layout the model was trained on.
    y_calib : array-like
        Multi-hot ground-truth labels of shape ``(n_samples, n_classes)``.
    device : str, optional
        Device used for inference (default ``"cuda"``).
    alpha : list of float, optional
        Target FNR risk levels (default ``[0.10]``).
    lambda_step_sz : float, optional
        Step size over the ``lambda`` grid in ``[0, 1]`` (default ``0.05``).
    batch_sz : int, optional
        Inference batch size (default ``32``).
    plot : bool, optional
        If True, plot the FNR risk curve with selected thresholds.

    Returns
    -------
    dict
        Per-threshold loss distributions, set sizes, and risks, plus
        ``"calib"``: a list of ``(alpha, lambda_thresh)`` pairs.
    """
    model.eval()
    x_calib_batched = torch.tensor(x_calib).type(torch.float)
    x_calib_batched = torch.split(x_calib_batched, batch_sz)
    pred_probs = []
    for batch in x_calib_batched:
        _, probs, _ = model(batch.to(device))
        pred_probs.append(probs.detach().cpu().numpy())
    pred_probs = np.concatenate(pred_probs, axis=0)

    FNR_dict = {}
    FNR_risk_list = []
    lambda_steps = np.arange(0, 1 + lambda_step_sz, lambda_step_sz)
    for i in lambda_steps[:-1]:
        pred_sets = np.uint16(pred_probs > (1.0 - i))
        ytrue_pred_intersect = pred_sets & y_calib
        # Tiny offset avoids /0 on empty-label rows (all-negative multi-hot).
        denom = np.sum(y_calib, axis=-1) + 1e-12
        FNR_loss = 1 - (np.sum(ytrue_pred_intersect, axis=-1) / denom)
        FNR_risk = np.mean(FNR_loss).item()
        FNR_dict[1.0 - i] = {
            "loss_dist": FNR_loss.tolist(),
            "set_size": np.sum(pred_sets, axis=1).tolist(),
            "risk": FNR_risk
        }
        FNR_risk_list.append(FNR_risk)
    # lambda = 1 includes every class, so FNR risk is 0
    FNR_risk_list.append(0)

    FNR_dict["calib"] = []
    for a in alpha:
        lambda_thresh = lambda_steps[np.array(FNR_risk_list) < a][0]
        FNR_dict["calib"].append((a, lambda_thresh))

    if plot:
        colors = plt.cm.viridis(np.linspace(0, 1, len(alpha)))
        plt.plot(lambda_steps, FNR_risk_list)
        plt.title("FNR Risk Calibration Curve")
        for i in range(len(alpha)):
            alpha_i = FNR_dict["calib"][i][0]
            lambda_i = FNR_dict["calib"][i][1]
            plt.axhline(
                alpha_i,
                linestyle="--",
                linewidth=2,
                label=fr"$\alpha_{{{i}}} = {alpha_i}$",
                color=colors[i],
            )
            plt.axvline(
                lambda_i,
                linestyle="--",
                linewidth=2,
                label=fr"$\lambda_{{{i}}} = {lambda_i}$",
                color=colors[i],
            )
        plt.xlabel("\u03BB(1 - confidence)")
        plt.ylabel("Risk")
        plt.legend()
        plt.show()
    return FNR_dict
