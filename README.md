# Self-Supervised Path Planning in Unstructured Environments via Global-Guided Differentiable Hard Constraint Projection

> This is a hard-constrained self-supervised deep learning training framework, aimed at utilizing hard-constrained neural networks to improve the determinism and reliability of learning-based algorithms in safety-critical scenarios.

## 🛠️ Installation

Before running the project, please ensure your environment meets the dependency requirements. You can install the necessary libraries with the following command:
```bash
pip install -r requirements.txt
```
## 🚀 Quick Start

The main workflow of this project strictly follows these three steps: Data Generation -> Pre-training -> Hard-constrained Training.

1. Data Generation
First, run the data generation script to prepare the training and testing datasets.

Default Path: Generated data will be automatically saved in the dataset/ folder within the current directory.

Custom Path: If needed, you can modify the save path configuration directly in the script.
```bash
python data_generator.py
```
2. Pre-training
Once the data is ready, use the pre-training script to initialize the model. This step allows the model to learn fundamental features, preparing it for the subsequent constraint-based training.
```bash
python pre_train.py
```
3. Hard-constrained Training
Finally, run the main training script. Based on the pre-trained model, this phase performs optimization with Hard Constraints.
```bash
python train.py
```
## 📂 File Structure

data_generator.py: Data generation script

pre_train.py: Pre-training script

train.py: Main training script with hard constraints

requirements.txt: List of project dependencies

dataset/: Default directory for data storage (created after running the generator)
