import argparse
import json
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from Calibration import FNR_Calibrator
from Modeling import Conv1D_RF_Classifier, Conv1D_RF_MultiClassifier
from Processor import POWDERRF_Processor
from train import evaluator, trainer


def parse_args():
    """Parse command-line arguments for train / evaluate / calibrate."""
    p = argparse.ArgumentParser(description="Train, calibrate, and evaluate RF fingerprint classifiers.")
    p.add_argument("--mode", nargs="+", choices=["train", "calibrate", "evaluate"], default=["train", "calibrate", "evaluate"])
    p.add_argument("--datadir", required=True, help="folder with all dataset subfolders")
    p.add_argument("--savepath", default="./powder_cochannelfingerprint_model")
    p.add_argument("--device", default='cuda')
    p.add_argument("--signal_type",choices=["Complex", "Real", "Imag", "Mag", "Phase", "fft", "IQ_All"] , default='fft')
    p.add_argument("--remove_tx", type=int, nargs="*", default=[], help="Transmitter indices to drop from files+labels.")
    p.add_argument("--must_have_tx", type=int, nargs="*", default=[], help="Transmitter indices that must be on.")
    p.add_argument("--min_tx", type=int, default=1, help="Minimum active transmitters. Default 1 if --task single else 0.")
    p.add_argument("--max_tx", type=float, default=6, help="Maximum active transmitters. Default 1 if --task single else inf.")
    p.add_argument("--samples_per_file", type=int, default=20000000, help="Number of samples in each .bin file")
    p.add_argument("--n_train_samples", type=int, default=25000)
    p.add_argument("--n_calib_samples", type=int, default=5000)
    p.add_argument("--n_test_samples", type=int, default=10000, help="Test size for paper per-gain eval (default: n-calib-samples).")
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--conf_thresh", type=str, default='0.50', help = "confidence thresholds for model classes as string can also be path so saved calibration dictionary" )
    p.add_argument("--batch_sz", type=int, default=256)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()





def main(args):

    os.makedirs(args.savepath, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    datadirs = [os.path.join(args.datadir, datadir) for datadir in 
                os.listdir(args.datadir) if "cache" not in datadir]

    try:
        conf_thresh = float(args.conf_thresh)
    except ValueError:
        with open(args.conf_thresh, 'r') as f:
            calib_dict = json.load(f)
            conf_thresh = 1 - calib_dict["calib"][0][1]

    if "train" in args.mode:
        print("Training....")
        model = Conv1D_RF_MultiClassifier(classes=args.classes)
        model.conf_thresh = conf_thresh
        Processor = POWDERRF_Processor(datadirs = datadirs, remove_tx = args.remove_tx, must_have_tx = args.must_have_tx , max_tx = args.max_tx, min_tx = args.min_tx)
        Processor.file_samples = args.samples_per_file
        xtrain, ytrain, _, _ = Processor(n_samples = args.n_train_samples, sample_len = 1024, signal_type = args.signal_type, train_test_split=1.0)
        model.train()
        model = trainer(model=model, xtrain=xtrain, ytrain=ytrain, batch_sz=args.batch_sz,
                      epochs=args.epochs, lr=args.lr, savepath=args.savepath, device=device)

    if "calibrate" in args.mode:
        print("Calibrating....")
        model = Conv1D_RF_MultiClassifier(classes=args.classes)
        ckpt = os.path.join(args.savepath, "final_model.pth")
        
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)

        Processor = POWDERRF_Processor(datadirs = datadirs, remove_tx = args.remove_tx, must_have_tx = args.must_have_tx , max_tx = args.max_tx, min_tx = args.min_tx)
        Processor.file_samples = args.samples_per_file
        xcalib, ycalib, _, _ = Processor(n_samples = args.n_calib_samples, sample_len = 1024, signal_type = args.signal_type, train_test_split=1.0)
        calib_dict = FNR_Calibrator(model, xcalib, ycalib, device=device, alpha=[args.alpha], lambda_step_sz=0.01, plot=False)
        conf_thresh = 1 - calib_dict["calib"][0][1]
        with open(os.path.join(args.savepath, "calib_results.json"), 'w') as f:
            json.dump(calib_dict, f)
        
    if "evaluate" in args.mode:
        print("Evaluating....")
        model = Conv1D_RF_MultiClassifier(classes=args.classes)
        ckpt = os.path.join(args.savepath, "final_model.pth")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model.conf_thresh = conf_thresh
        Processor = POWDERRF_Processor(datadirs = datadirs, remove_tx = args.remove_tx, must_have_tx = args.must_have_tx , max_tx = args.max_tx, min_tx = args.min_tx)
        Processor.file_samples = args.samples_per_file
        xtest, ytest, _, _ = Processor(n_samples = args.n_test_samples, sample_len = 1024, signal_type = args.signal_type, train_test_split=1.0)
        model.eval()
        eval_dict = evaluator(model=model, xtest=xtest, ytest=ytest, batch_sz=args.batch_sz, savepath=args.savepath, device=device)
   
   

if __name__ == "__main__":
    args = parse_args()
    main(args)
