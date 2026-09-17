from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas as pd

try:
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    SummaryWriter = None
    EventAccumulator = None


class _NullWriter:
    """No-op TensorBoard writer used when tensorboard is not installed."""

    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def trainer(
    model=None,
    xtrain=np.array([]),
    ytrain=np.array([]),
    batch_sz=100,
    epochs=100,
    lr=0.001,
    savepath="./",
    i_checkpoint=5000,
    device="cuda",
    val_split=0.05,
):
    """Train a classifier and log metrics to TensorBoard.

    The first ``val_split`` fraction of ``xtrain`` / ``ytrain`` is held out
    for validation (no extra shuffle). Checkpoints are written every
    ``i_checkpoint`` optimizer steps, and the final weights are saved as
    ``final_model.pth``.

    View logs with::

        tensorboard --logdir=<savepath>/tensorboard_logs

    Parameters
    ----------
    model : torch.nn.Module
        Model whose ``forward`` returns ``(logits, loss)`` in train mode and
        ``(logits, probs, preds)`` in eval mode.
    xtrain : numpy.ndarray
        Training features.
    ytrain : numpy.ndarray
        Training labels.
    batch_sz : int, optional
        Mini-batch size (default ``100``).
    epochs : int, optional
        Number of training epochs (default ``100``).
    lr : float, optional
        Adam learning rate (default ``0.001``).
    savepath : str, optional
        Directory for checkpoints and TensorBoard logs.
    i_checkpoint : int, optional
        Optimizer-step interval between checkpoints (default ``25``).
    device : str, optional
        Device used for training (default ``"cuda"``).
    val_split : float, optional
        Fraction of the provided training arrays used for validation.
        Set to ``0`` to skip validation.

    Returns
    -------
    torch.nn.Module
        The trained model (left in train mode).
    """
    if savepath[-1] != "/":
        savepath += "/"
    if SummaryWriter is None:
        print("tensorboard is not installed; training metrics will not be logged. pip install tensorboard")
        writer = _NullWriter()
    else:
        writer = SummaryWriter(log_dir=savepath + "tensorboard_logs")

    if val_split != 0:
        val_index = int(np.floor(xtrain.shape[0] * val_split))
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model = model.to(device)
    loss_sum = 0
    i = 0
    for epoch in range(epochs):
        print("Epoch:", epoch)
        for x_batch, y_batch in tqdm(zip(xtrain_batched, ytrain_batched)):
            i += 1
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            _, loss = model(data_in=x_batch, y_true=y_batch)
            loss.backward()
            loss_sum += loss.item()
            avg_loss = loss_sum / i

            writer.add_scalar("Loss/batch", loss.item(), i)
            writer.add_scalar("Loss/avg", avg_loss, i)

            if i % i_checkpoint == 0:
                model_savepath = savepath + "checkpoint-" + str(i) + ".pth"
                torch.save(model.state_dict(), model_savepath)
            optimizer.step()

        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
        print(f"Loss/epoch_avg:{avg_loss}")
        if val_split != 0:
            model.eval()
            with torch.no_grad():
                preds = []
                pred_probs = []
                for batch in xval_batched:
                    batch = batch.to(device)
                    _, probs, pred = model(batch)
                    pred_probs.append(probs.detach().cpu().numpy())
                    preds.append(pred.detach().cpu().numpy())
                preds = np.concatenate(preds, axis=0)
                pred_probs = np.concatenate(pred_probs, axis=0)
                acc_score = accuracy_score(preds.flatten(), yval.flatten())
                roc_auc = roc_auc_score(yval, pred_probs, multi_class="ovr")
                writer.add_scalar("Metrics/roc_auc", roc_auc, epoch)
                writer.add_scalar("Metrics/acc", acc_score, epoch)
                print(f"Metrics/roc_auc:{roc_auc}")
                print(f"Metrics/acc:{acc_score}")
            model.train()

    model_savepath = savepath + "final_model.pth"
    torch.save(model.state_dict(), model_savepath)
    return model


def evaluator(model=None, xtest=np.array([]), ytest=np.array([]), batch_sz=32, savepath="./", device="cuda", plot = True):
    """Evaluate a trained classifier and write a classification report.

    Computes accuracy, a per-class classification report, and a micro-averaged
    ROC curve. The report is saved to ``savepath/results.json``.

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval-compatible form. ``model(x)`` must return
        ``(logits, probs, preds)``.
    xtest : numpy.ndarray
        Test features.
    ytest : numpy.ndarray
        Test labels (multi-hot or one-hot).
    batch_sz : int, optional
        Inference batch size (default ``32``).
    savepath : str, optional
        Directory used to write ``results.json``.
    device : str, optional
        Device used for inference (default ``"cuda"``).

    Returns
    -------
    dict
        Accuracy, micro ROC coordinates, AUROC, and the classification report.
    """
    model = model.to(device)
    xtest_batched = torch.tensor(xtest).type(torch.float)
    xtest_batched = torch.split(xtest_batched, batch_sz)
    preds = []
    pred_probs = []
    for batch in tqdm(xtest_batched):
        batch = batch.to(device)
        _, probs, pred = model(batch)
        pred_probs.append(probs.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    pred_probs = np.concatenate(pred_probs, axis=0)
    report = classification_report(ytest, preds, output_dict=True, target_names=range(len(ytest[0])))
    fpr_micro, tpr_micro, _ = roc_curve(ytest.flatten(), pred_probs.flatten())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    acc = accuracy_score(ytest.flatten(), preds.flatten())

    print(f"Metrics/acc:{acc}")
    print(report)
    if plot:
        plt.plot(fpr_micro, tpr_micro)
        plt.text(
            0.30,
            0.95,
            f"AUROC = {roc_auc_micro:.4f}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color="blue",
        )
        plt.show()

    results = {
        "acc": acc,
        "fpr_micro": fpr_micro.tolist(),
        "tpr_micro": tpr_micro.tolist(),
        "roc_auc": roc_auc_micro,
        "report": report,
    }
    results_path = savepath + "eval_results.json"
    with open(results_path, "w") as f:
            json.dump(results, f)
    
    return results


def load_tensorboard_logs(log_dir):
    """Load scalar time series from a TensorBoard log directory.

    Parameters
    ----------
    log_dir : str
        Path to a TensorBoard event directory, e.g.
        ``"<savepath>/tensorboard_logs"``.

    Returns
    -------
    dict of pandas.DataFrame
        One frame per scalar tag, with columns ``wall_time``, ``step``,
        and ``value``.
    """
    if EventAccumulator is None:
        raise ImportError("load_tensorboard_logs requires tensorboard. pip install tensorboard")
    ea = EventAccumulator(log_dir)
    ea.Reload()

    available_tags = ea.Tags()["scalars"]
    print("Available tags:", available_tags)

    all_data = {}
    for tag in available_tags:
        events = ea.Scalars(tag)
        all_data[tag] = pd.DataFrame(events)
    return all_data
