## Cascade Attention Guided Residue GAN for Cross-Modal Translation
### Requirements
- pytorch 0.4+
- python 3.6.8
- visdom 0.1.8.8
- pillow 5.4.1
### Model
Download the model via [google drive](https://drive.google.com/open?id=1k6rne0ebJmcJIpgSwsCqlEMffTeEaKKa) and put models into checkpoints dir.
### Train
`$ sh ltrain.sh`
### Test
`$ sh ltest.sh`
### Acknowledgment
Our code borrows heavily from the the pix2pix implementation [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/).
