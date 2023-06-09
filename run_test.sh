python3 test.py \
--baseroot './mask/alarm_v2/img' \
--baseroot_mask './mask/alarm_v2/mask' \
--results_path './mask/alarm_v2' \
--gan_type 'WGAN' \
--gpu_ids '7' \
--multi_gpu True \
--epoch 20 \
--batch_size 1 \
--num_workers 1 \
--pad_type 'zero' \
--activation 'elu' \
--norm 'none' \
