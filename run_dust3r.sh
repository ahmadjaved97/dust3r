#!/bin/bash

torchrun --nproc_per_node=8 train.py \
  --train_dataset "FreiburgDatasetThermal(split='train', root_dir='/lustre/mlnvme/data/s63ajave_hpc-cuda_lab', dataset_filename='dataset_v1_224_train.json', resolution=224, use_rgb=False, use_enhance=False)" \
  --test_dataset "FreiburgDatasetThermal(split='test', root_dir='/lustre/mlnvme/data/s63ajave_hpc-cuda_lab', dataset_filename='dataset_v1_224_test.json', resolution=224, seed=777, use_rgb=False, use_enhance=False)" \
  --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
  --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
  --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
  --pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
  --lr 0.0001 \
  --min_lr 1e-06 \
  --warmup_epochs 10 \
  --epochs 100 \
  --batch_size 16 \
  --accum_iter 1 \
  --save_freq 5 \
  --keep_freq 10 \
  --eval_freq 1 \
  --output_dir "/lustre/mlnvme/data/s63ajave_hpc-cuda_lab/checkpoints/dust3r_freiburg_224_thermal8"
