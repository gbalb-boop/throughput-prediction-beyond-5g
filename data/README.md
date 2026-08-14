# Dataset

This project uses the dataset associated with:

T. Tsourdinis, I. Chatzistefanidis, N. Makris, T. Korakis, N. Nikaein, and S. Fdida, **"Service-aware real-time slicing for virtualized beyond 5G networks,"** *Computer Networks*, vol. 247, 110445, 2024.

The dataset is available from the original project repository:

`https://github.com/teo-tsou/app_aware_5g/tree/master/dataset`

## Setup

Download:

```text
ue-lte-network-traffic-stats.csv
```

and place it in this directory:

```text
data/ue-lte-network-traffic-stats.csv
```

Then run:

```bash
python src/data_preprocessing.py
```

The preprocessing script generates the chronological training, validation, and testing datasets used by the models.

The generated CSV files are intentionally excluded from version control because they can be reproduced from the original dataset.
