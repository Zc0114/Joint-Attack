# Federated Learning Security Experiments

This repository accompanies the manuscript *Synergizing Backdoor and Membership Inference Attacks via Dynamic Triggers in Federated Learning*. It contains Python research code for evaluating model integrity and training-data privacy in simulated federated image classification.

## Experiment suites

| Directory | Dataset | Classifier |
| --- | --- | --- |
| `1.cifar10-alexnet-1/` | CIFAR-10 | AlexNet |
| `2.fmnist-cnn-1/` | Fashion-MNIST | CNN |
| `3.cifar100-alexnext-1/` | CIFAR-100 | AlexNet |

## Repository layout

Each suite contains a `requirements.txt` file, YAML configuration files, and Python source code. The `src/` directory contains the model, client, server, and utility modules. The CIFAR-100 suite also includes additional YAML files.

## Dependencies

The three suites list the same dependencies: NumPy, Matplotlib, Pillow, PyYAML, scikit-learn, SciPy, PyTorch, torchvision, and tqdm. The requirements files list package names without version pins.

## Results and terminology

The manuscript reports classification accuracy on the main task (MA), backdoor attack success rate (BA-ASR), membership inference attack success rate (MIA-ASR), and area under the ROC curve for membership inference (MIA-AUC). Consult the manuscript for the experimental comparisons, ablation results, and evaluation of defenses.

## Citation

When referencing this repository, cite the accompanying manuscript using the author names and publication details from the version you consulted.
