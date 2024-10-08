# CGAN Training for Time-Series Data

This project implements a Conditional Generative Adversarial Network (CGAN) to generate synthetic time-series data using PyTorch. The CGAN is designed to work with time-series data and includes components for data preprocessing, model training, and evaluation.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Description

This project uses a CGAN to generate time-series data. It involves:
- Loading and preprocessing data.
- Training a generator and discriminator with specific loss functions.
- Saving model checkpoints and logs.
- Plotting the losses and generated samples during training.

## Installation

To set up this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/katarzynamichalowska/windturbine_loads.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd windturbine_loads
    ```

3. **Install dependencies:**

    Create a virtual environment and install the required packages. You can use `requirements.txt` if available or install packages manually.

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the CGAN, execute the training script:

```bash
python train_cgan.py
```

## Contact

Contact person: [katarzyna.michalowska@sintef.no](mailto:katarzyna.michalowska@sintef.no)
