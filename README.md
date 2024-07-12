# STCNet
Integrating Spatio-Temporal Dynamics with Subject-Aware Contrastive (SAC) Learning for Hand Gesture Recognition


### data process

first, download dataset (NinaPro DB1, DB2, DB4)
https://ninapro.hevs.ch/

second, you have to run denosing.m file!

finally
```
python emg_process.py --dataset nina1 --path 'your dataset folder path'
```

### how to run train file

```
python train_sac.py --dataset nina1 --batch_size 1024 --cosine --gamma 0.0
python train_ce.py --dataset nina1 --cosine --encoder "put your result pth file from train_multi.py"
```

if you run train file on nina2 or nina4, you add '--sampled' argumentation!

### how to test

```
python test.py --dataset nina2 --sampled --model_path 'put result pth file'
```

if you run test file on nina2 or nina4, you add '--sampled' argumentation, too!
