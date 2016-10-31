pkill -9 python

python a3c_dist.py \
     --ps_hosts=localhost:2226 \
     --worker_hosts=localhost:2227,localhost:2228,localhost:2229 \
     --job_name=ps --task_index=0 \
     --env Pong-v0 \
     --diff_frame True --preprocess True --inp_dim 6400 --out_dim 2 \
     --actions 2,3 & \
python a3c_dist.py \
     --ps_hosts=localhost:2226 \
     --worker_hosts=localhost:2227,localhost:2228,localhost:2229 \
     --job_name=worker --task_index=0 \
     --env Pong-v0 \
     --diff_frame True --preprocess True --inp_dim 6400 --out_dim 2 \
     --actions 2,3 \
      > pong_w1.txt & \
python a3c_dist.py \
     --ps_hosts=localhost:2226\
     --worker_hosts=localhost:2227,localhost:2228,localhost:2229 \
     --job_name=worker --task_index=1 \
     --env Pong-v0 \
     --diff_frame True --preprocess True --inp_dim 6400 --out_dim 2 \
     --actions 2,3 \
     > pong_w2.txt & \
python a3c_dist.py \
     --ps_hosts=localhost:2226\
     --worker_hosts=localhost:2227,localhost:2228,localhost:2229 \
     --job_name=worker --task_index=2 \
     --env Pong-v0 \
     --diff_frame True --preprocess True --inp_dim 6400 --out_dim 2 \
     --actions 2,3 \
     > pong_w3.txt