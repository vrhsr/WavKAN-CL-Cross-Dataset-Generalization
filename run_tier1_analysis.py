import subprocess
import os

models = ['wavkan', 'resnet', 'vit', 'spline_kan', 'mlp', 'dann']
scripts = ['src.compute_mmd', 'src.test_forgetting']

for model in models:
    ckpt = f'experiments/{model}_seed42.pth'
    if not os.path.exists(ckpt):
        ckpt = f'experiments/{model}_endpoint.pth'
        
    for script in scripts:
        cmd = f"python -m {script} --model {model} --checkpoint {ckpt}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)
