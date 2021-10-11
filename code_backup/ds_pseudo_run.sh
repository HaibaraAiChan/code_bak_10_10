#!/bin/bash

# File=products_pseudo_pure_time_block_generate.py
File=products_pseudo_final_version.py


# reddit dataset 
batch_size=(153431 76716 38358 19179 9590 4795 2400 1200)
for i in ${batch_size[@]};do
  python $File \
  --dataset reddit \
  --aggre mean \
  --selection-method range \
  --batch-size $i \
  --num-epochs 6 \
  --eval-every 5 > range/reddit_mean_pseudo_log/bs_${i}_6_epoch.log
# done
# for i in ${batch_size[@]};do
  python $File \
  --dataset reddit \
  --aggre mean \
  --selection-method random \
  --batch-size $i \
  --num-epochs 6 \
  --eval-every 5 > random/reddit_mean_pseudo_log/bs_${i}_6_epoch.log
done



# # ogbn-protcuts
# batch_size=(196571 98308 49154 24577 12289 6245 3000 1500)
# for i in ${batch_size[@]};do
#   python $File \
#   --dataset ogbn-products \
#   --aggre lstm \
#   --selection-method range \
#   --batch-size $i \
#   --num-epochs 6 \
#   --eval-every 5 > range/products_lstm_pseudo_log/bs_${i}_6_epoch.log
# # done
# # for i in ${batch_size[@]};do
#   python $File \
#   --dataset ogbn-products \
#   --aggre lstm \
#   --selection-method random \
#   --batch-size $i \
#   --num-epochs 6 \
#   --eval-every 5 > random/products_lstm_pseudo_log/bs_${i}_6_epoch.log
# done

# for i in ${batch_size[@]};do
#   # python products_pseudo_pure_.py --dataset ogbn-products --aggre mean --selection-method range --batch-size $i --num-epochs 6 --eval-every 5 > random/products_mean_pseudo_log/bs_${i}_6_epoch.log
#   python products_pseudo_pure_.py --dataset ogbn-products --aggre lstm --selection-method range --batch-size $i --num-epochs 6 --eval-every 5 > range/products_lstm_pseudo_log/bs_${i}_6_epoch.log
#   python products_pseudo_pure_.py --dataset ogbn-products --aggre lstm --selection-method random --batch-size $i --num-epochs 6 --eval-every 5 > random/products_lstm_pseudo_log/bs_${i}_6_epoch.log
# done
