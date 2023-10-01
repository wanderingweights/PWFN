# Probabilistic Weight Fixing: Large-scale Training of Neural Network Weight Uncertainties for Quantization (PWFN)

## Introduction

Weight-sharing quantization is a method devised to curtail energy consumption during inference in expansive neural networks by binding their weights to a limited set of values. This repository presents an implementation of a pioneering probabilistic framework anchored in Bayesian neural networks (BNNs) that emphasizes the distinct role of weight position. This approach, accepted for presentation at NeurIPS 2023, exhibits enhanced noise resilience and downstream compressibility, outstripping performance across multiple architectures.

You can access the paper on [arXiv](https://arxiv.org/abs/2309.13575).

## Key Features

- Probabilistic framework anchored in Bayesian neural networks (BNNs).
- Variational relaxation for discerning weight movement to cluster centers.
- Novel initialization setting and regularization term for adept BNN training.
- Iterative clustering method for superior compressibility.

## Quick Start

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install torch==1.13.1+cu111 torchvision==0.14.1+cu111 torchmetrics==0.11.3 timm==0.6.12 lightning==1.9.4 numpy==1.23.5 pandas==1.5.3 scipy==1.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

### Clone the Repository

```bash
git clone https://github.com/subiawaud/PWFN.git
cd PWFN
```

### Running the Experiments

Our experiments are conducted on the ImageNet dataset with a variety of models, including but not limited to: ResNets-(18,34,50), DenseNet-161, and the challenging DeiT (small and tiny). The implementation is designed to be flexible, allowing any model from the Timm library to be used as the `<chosen_model>`.

To initiate the experiments:

```bash
python main.py --lr <learning_rate> --start_epochs <initial_epochs> --rest_epochs <subsequent_epochs> --reg <regularization> --b <b_value> --data <dataset_name> --start_sigma <initial_sigma> --end_sigma <final_sigma> --reg_function <regularization_function> --data_loc /path/to/dataset --model <chosen_model>
```

**Variables Explanation:**
- `<learning_rate>`: Learning rate for the optimizer (e.g., 0.001).
- `<initial_epochs>`: Number of epochs for the initial training phase (e.g., 3).
- `<subsequent_epochs>`: Number of epochs for the subsequent training phases (e.g., 3).
- `<regularization>`: Regularization value (e.g., 1.0).
- `<b_value>`: B-value parameter used in the method (e.g., 7).
- `<dataset_name>`: Name of the dataset used for the experiment (e.g., 'imagenet').
- `<initial_sigma>`: Initial sigma value for the weight distributions (e.g., 1).
- `<final_sigma>`: Final sigma value for the weight distributions (e.g., 0.5).
- `<regularization_function>`: Regularization function to be used (e.g., 'linear').
- `<chosen_model>`: Desired model architecture from the Timm library, such as `resnet18`, `resnet34`, `resnet50`, `densenet161`, `deit_small`, or `deit_tiny`.

### Results

Performance metrics and results will be stored in specified paths within the script for further evaluation.

## Abstract

Weight-sharing quantization is an innovative technique targeting energy reduction during neural network inference. Our proposed probabilistic framework, grounded in Bayesian neural networks (BNNs), alongside a variational relaxation, surpasses contemporary techniques in both compressibility and accuracy across diverse architectures. The work has been accepted for presentation at NeurIPS 2023.

## Citing

If our work proves instrumental in your research, please consider citing:

```bibtex
@article{subia2023probabilistic,
  title={Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantization},
  author={Subia-Waud, Christopher and Dasmahapatra, Srinandan},
  journal={NeurIPS 2023},
  year={2023}
}
```

## Acknowledgements

This research is attributed to the School of Electronics & Computer Science, University of Southampton, UK.

## License

The project is licensed under the MIT License. For detailed information, refer to [LICENSE.md](LICENSE.md).

## Contact

For any questions or feedback, contact [Christopher Subia-Waud](mailto:cc2u18@soton.ac.uk) or [Srinandan Dasmahapatra](mailto:sd@soton.ac.uk), or simply raise an issue on this GitHub repository.

---
