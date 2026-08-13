# Model weights

## Project-specific fall detector

The repository includes the validated GMDCSA24 PoseC3D checkpoint:

```text
experiments/outputs/posec3d_gmdcsa24/best_acc_top1_epoch_45.pth
```

- Model: PoseC3D SlowOnly-R50, binary ADL/Fall classifier
- Selection: best validation top-1 checkpoint at epoch 45
- File size: 8,197,226 bytes
- SHA-256: `7d60a2f502d0f83e401b3017f2a416d193ba2f9c3b4432925678108b2bac3a0a`
- Evaluation reference: `experiments/outputs/losocv/summary.md`

The detector, pose-estimation, and PoseC3D pretraining checkpoints under
`models/pretrained/` are third-party dependencies and remain excluded from
the repository. Use the existing download/configuration scripts to obtain
those files locally.
