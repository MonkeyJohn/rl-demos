pkill -9 python

python a3c_dist.py \
     --ps_hosts=localhost:3226 \
     --worker_hosts=localhost:3227,localhost:3228 \
     --job_name=ps --task_index=0 \
     --env CartPole-v0 \
     --diff_frame False --preprocess False --inp_dim 4 --out_dim 2 \
     --actions 0,1 & \
python a3c_dist.py \
     --ps_hosts=localhost:3226 \
     --worker_hosts=localhost:3227,localhost:3228 \
     --job_name=worker --task_index=0 \
     --env CartPole-v0 \
     --diff_frame False --preprocess False --inp_dim 4 --out_dim 2 \
     --actions 0,1 > cp_w1.txt & \
python a3c_dist.py \
     --ps_hosts=localhost:3226\
     --worker_hosts=localhost:3227,localhost:3228 \
     --job_name=worker --task_index=1 \
     --env CartPole-v0 \
     --diff_frame False --preprocess False --inp_dim 4 --out_dim 2 \
     --actions 0,1 > cp_w2.txt