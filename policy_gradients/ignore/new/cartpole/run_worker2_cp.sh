#!/bin/bash
python pg_dist_cp.py \
     --ps_hosts=localhost:5226\
     --worker_hosts=localhost:5227,localhost:5228\
     --job_name=worker --task_index=1
