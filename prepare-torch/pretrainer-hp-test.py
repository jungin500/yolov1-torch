import subprocess
import random
import os

# Hyperparameter test script
# 2021/06/15 LimeOrangePie.

if __name__ == '__main__':
    subprocess_env = os.environ.copy()
    subprocess_env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    epochs = 100

    lr_search_space = [0.1, 0.0001]

    lr_volume = abs(lr_search_space[1] - lr_search_space[0])
    lr_margin = min(lr_search_space[0], lr_search_space[1])

    descending = True

    for lr_int in range(epochs):
        if descending:
            lr_float = (lr_margin + lr_volume) - (lr_int / epochs * lr_volume)
        else:
            lr_float = lr_margin + (lr_int / epochs * lr_volume)
        run_name = "HyperparameterTest"

        print("Try %d of %d: learning rate %.5f, one epochs each" % (lr_int, epochs, lr_float))
        # EnvVar for deterministic CUDA algorithms
        result = subprocess.run(('python pretrainer.py -g -b 64 -e 1 -p 16 -l %.5f -rn %s --seed 123456 --limit-batch' % (lr_float, run_name)).split(' '),
            env=subprocess_env,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
        )
        if result.returncode != 0:
            print("Error - exited abnormally")