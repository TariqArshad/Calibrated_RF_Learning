# Calibrated RF Learning

## Dataset

 [POWDER Co-Channel Protocol dataset](https://huggingface.co/datasets/T-Arshad/POWDER_CoChannel_Protocol_Dataset): 768 indoor OTA captures (802.11a, 4G LTE, 5G NR) from the POWDER testbed. Each `Round*_Gain*` folder holds 64 `.bin` / `.json` pairs.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="T-Arshad/POWDER_CoChannel_Protocol_Dataset",
    repo_type="dataset",
    allow_patterns=["Round1_Gain90/*"],  # drop this line / broaden patterns for all ~123 GB
    local_dir="POWDER_CoChannel_Protocol",
)
```

## Setup using conda 

```bash
conda create -n calibrated_rf_env python=3.10
conda activate calibrated_rf_env
pip install -r requirements.txt
```

## Run

```bash
python src/Calibrated_RF_Learning/main.py --mode train calibrate evaluate \
  --datadir POWDER_CoChannel_Protocol/
  --savepath ./powder_cochannelfingerprint_model
```

## Citation

```bibtex
@inproceedings{abdulquddoos2026calibrated,
  title={Calibrated {RF}-Fingerprinting Under Interference With Heterogeneous Transmission Protocols},
  author={Abdul-Quddoos, Tariq and Li, Xiangfang and Qian, Lijun},
  booktitle={IEEE International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)},
  year={2026},
  organization={IEEE}
}
```

MIT License.
