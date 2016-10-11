#!/bin/bash
rm -rf tmp_cp
pkill -9 python
python pg_dist_cp.py \
     --ps_hosts=localhost:5226\
     --worker_hosts=localhost:5227,localhost:5228\
     --job_name=ps --task_index=0
