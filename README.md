# Learning State-Based Node Representations from a Class Hierarchy for Fine-Grained Open-Set Detection

This repository is build using https://github.com/KMnP/vpt, please refer to this repository for environment setup.

For datasets, please download the respective dataset from the source. Store the datasets to respective folders:


- D_ALL/AWA2
- D_ALL/TINYIMAGENET
- D_ALL/CUB
- D_ALL/CARS
- D_ALL/CIFAR100

The taxonomy of each dataset is given in the respective folder. Closed-set and open-set classes are provided for each dataset. Please make sure to store training images to `D_ALL/<dataset>/train` and test samples from closed-set classes to `D_ALL/<dataset>/known` and openset classes to `D_ALL/<dataset>/novel`.

Run the following code to get started.

```
 bash run_locally.sh
``` 

For the config, make sure the right config is passed.

- For CUB, use ` --config-file configs/prompt/cub_al.yaml`
- For TINYIMAGENET, use ` --config-file configs/prompt/tiny_al.yaml`

Similarly, make sure to pass correct config for other datasets as well.

After the training is complete, the logits are saved for closed set and open set samples. Run the `scripts/check_logit.py`  file to get OA@50 and AUC.
