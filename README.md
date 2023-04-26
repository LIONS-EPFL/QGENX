# Code for the paper [Distributed Extra-Gradient With Optimal Complexity And Communication Guarantees](https://infoscience.epfl.ch/record/300852), published at ICLR2023

The training code is a modified version of [ this repo by Gidel et al.](https://github.com/GauthierGidel/Variational-Inequality-GAN), with the vendored [torch_cgx](./torch_cgx) version being included since the currently public version uses a different API (we are working on releasing a version which uses this new API and will include improved quantization as well).

In order to replicate the experiments in our paper

1. Create a file `wandbkey` with `export WANDB_API_KEY=you_key_here` in order to enable logging the outputs
2. spin up 3 V100 GPU nodes with your favourite kubernetes provider and adapt the `kubelaunch.sh`file as needed
3. then launch with `NUM_PODS=3 zsh stilaunch.sh` (or bash, or fish etc.)

By default, this will

- delete the previous app, if it exists
- call the `build_image_local.sh`, which will not do anything by default (commented out), but you can adopt it in case you want to update the image
- apply the `gpu3.yaml` to your namespace to recreate the app with the image created in the `build_image_local.sh` step
- enter a loop as it waits for the pods to come online
- tar.gzip the experiment files, upload them to the arbitrarily selected head node, perform some hackery to enable you to directly ssh into each node and to ensure mpi will be able to ssh as well (this is what the `qgqg_ed25519.pub` and `qgqg_ed25519` files are for, we didn't in fact leak our own SSH keys :-)
- connect you to the head node, dropping you into a tmux session

Here, you can edit the `dist_mpi_launch.sh` to tweak hyperparameters or simply select between the "FULL_CMD","UNIFORM_CMD" or (coming soon) "NUQ_CMD" in order to run the experiments. Then simply detach and get the results in wandb.
Once done, you can run `delete.sh` to destroy the app.

# Updating the dockerfile

For now we provide an image on our own dockerhub account, but this might change in the future. You can retarget the build with the instructions below.

In order to update the dockerfile, you will need to enter the `torch_cgx` directory and run `docker build -t qgeg_cgx:august`, then enter the root update the `build_image_local.sh`, uncomment it, add your own docker org as push target and run it (or `docker build -t qgeg_cgx:august` for manual debugging).
You can then update the `gpu3.yaml` to pull from your own docker org and you are set.
