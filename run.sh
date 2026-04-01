export DATA_ROOT='/data/datasets/$USER/ITML604_DL_Pretrained_MnasNet/content/imagenette2-160'


python3 mnasnet_imagenette_simple_split.py \
   --data_root $DATA_ROOT \
   --out_dir runs_mnasnet_T4_test1 \
   --weights imagenet \
   --img_size 224 \
   --epochs 8 \
   --batch_size 128 \
   --lr 3e-4

# python3 mnasnet_imagenette_kfold.py \
#     --data_root $DATA_ROOT \
#     --out_dir runs_mnasnet_T4_test1 \
#     --weights imagenet \
#     --img_size 224 \
#     --epochs 8 \
#     --batch_size 128 \
#     --lr 3e-4 \
#     --kfold 5

