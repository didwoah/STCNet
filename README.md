# MSC-HGR
Multi-Supervised Contrastive Learning for sEMG Hand Gesture Recognition


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
python train_multi.py --dataset nina1 --batch_size 1024 --cosine --gamma 0.1
python train_ce.py --dataset nina1 --cosine --encoder "put your result pth file from train_multi.py"
```
