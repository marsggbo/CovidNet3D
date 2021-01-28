srun -n 1 --cpus-per-task 2 python retrain.py \
--config_file outputs/checkpoint/version_0/search_ct.yaml \
--arc_path outputs/checkpoint/version_0/epoch_0.json  \
input.size [128,128]