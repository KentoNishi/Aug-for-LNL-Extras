# Aug-for-LNL-Extras

Extra bits of disorganized code for plotting, training, etc. related to our CVPR 2021 paper ["Augmentation Strategies for Learning with Noisy Labels"](https://github.com/KentoNishi/Augmentation-for-LNL).

## Cloning

This repository uses submodules. Make sure you clone with the `--recursive` flag.

```bash
git clone git@github.com:KentoNishi/Aug-for-LNL-Extras.git --recursive
```

Scripts and notebooks are only published for demonstration purposes only. **There is no guarantee that they will run on your local machine**.
> Some of these scripts even contain absolute paths specific to our own machines!

## Some useful commands
**(for my own use)**

### Save per-sample loss at each epoch as a tensor (for plotting)
```bash
cd Augmentation-for-LNL
python train_cifar.py --preset c10.50sym.AugDesc-WS --machine extras
```
