# Probabilistic Weight Fixing: Large-scale Training of Neural Network Weight Uncertainties for Quantization (PWFN)

## Introduction

Weight-sharing quantization is a method devised to curtail energy consumption during inference in expansive neural networks by binding their weights to a limited set of values. This repository presents an implementation of a pioneering probabilistic framework anchored in Bayesian neural networks (BNNs) that emphasizes the distinct role of weight position. This approach, accepted for presentation at NeurIPS 2023, exhibits enhanced noise resilience and downstream compressibility, outstripping performance across multiple architectures.

You can access the paper on [arXiv](https://arxiv.org/abs/2309.13575).

## Key Features

Key Features

* Consideration of Weight Position: Emphasis on the role of weight position in determining weight movement during quantization.
* Initialization & Regularization: Introduction of a specific initialization setting and a regularization term, facilitating the training of BNNs on extensive datasets and model combinations.
* Noise-Tolerance Guidance: Use of learned sigma terms to assist in network compression decisions.
* Improved Compressibility & Accuracy: Enhanced compressibility and accuracy observed across various architectures, including ResNet models and transformer-based designs.
*   Results with DeiT-Tiny: Notable accuracy improvement on ImageNet with a quantized DeiT-Tiny, representing its weights with fewer unique values.

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
python wfn_bayes.py --lr <learning_rate> --start_epochs <initial_epochs> --rest_epochs <subsequent_epochs> --reg <regularization> --b <b_value> --data <dataset_name> --start_sigma <initial_sigma> --end_sigma <final_sigma> --reg_function <regularization_function> --data_loc /path/to/dataset --model <chosen_model> --prior <prior> --sigma_join <sigma_join>
```

**Variables Explanation:**
- `<learning_rate>`: Learning rate for the optimizer (e.g., 0.001).
- `<initial_epochs>`: Number of epochs for the initial training phase (e.g., 1).
- `<subsequent_epochs>`: Number of epochs for the subsequent training phases (e.g., 3).
- `<regularization>`: Regularization value (0.0004882812 in the paper)
- `<b_value>`: B-value parameter used in the method (e.g., 7).
- `<dataset_name>`: Name of the dataset used for the experiment (e.g., 'imagenet', 'cifar10').
- `<initial_delta>`: Initial delta value for the weight distributions (e.g., 1, in the paper initial = final)
- `<final_delta>`: Final delta value for the weight distributions
- `<regularization_function>`: Regularization function to be used (e.g., 'linear').
- `<chosen_model>`: Desired model architecture from the Timm library, such as `resnet18`, `resnet34`, `resnet50`, `densenet161`, `deit_small`, or `deit_tiny`.
- `<prior>`: Do we apply the prior initialisation based on pow2 distances
-  `<sigma_join>`: What method do we use to aggregate sigmas after a weight is clustered `std_mu` is used in the paper other options include `allow_retraining` which leaves the sigmas as they are and allows them to continue to change after the mu value is fixed and `keep_the_same_divide_by_10` which simply makes the sigma smaller by a factor of 10. 


To replicate the paper experiment settings -> 


1. For model `deit_small_patch16_224`:

```bash
python wfn_bayes.py  --model deit_small_patch16_224 --reg_function linear --data imagenet --lr 0.001 --start_epochs 1 --rest_epochs 3 --reg 0.00048828125 --start_delta 1.0 --end_delta 1.0 --inc 2 --b 7 --sigma_join std_mu --want_to_save --prior --zero_fix
```

2. For model `deit_tiny_patch16_224`:

```bash
python wfn_bayes.py  --model deit_tiny_patch16_224 --reg_function linear --data imagenet --lr 0.001 --start_epochs 1 --rest_epochs 3 --reg 0.00048828125 --start_delta 1.0 --end_delta 1.0 --inc 2 --b 7 --sigma_join std_mu --want_to_save --prior --zero_fix
```

3. For model `resnet18`:

```bash
python wfn_bayes.py  --model resnet18 --reg_function linear --data imagenet --lr 0.001 --start_epochs 1 --rest_epochs 3 --reg 0.00048828125 --start_delta 1.0 --end_delta 1.0 --inc 2 --b 7 --sigma_join std_mu --want_to_save --prior --zero_fix
```

4. For model `resnet34`:

```bash
python wfn_bayes.py  --model resnet34 --reg_function linear --data imagenet --lr 0.001 --start_epochs 1 --rest_epochs 3 --reg 0.00048828125 --start_delta 1.0 --end_delta 1.0 --inc 2 --b 7 --sigma_join std_mu --want_to_save --prior --zero_fix
```

5. For model `resnet50`:

```bash
python wfn_bayes.py  --model resnet50 --reg_function linear --data imagenet --lr 0.001 --start_epochs 1 --rest_epochs 3 --reg 0.00048828125 --start_delta 1.0 --end_delta 1.0 --inc 2 --b 7 --sigma_join std_mu --want_to_save --prior --zero_fix
```

---


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
  journal={arXiv preprint arXiv:2309.13575},
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
