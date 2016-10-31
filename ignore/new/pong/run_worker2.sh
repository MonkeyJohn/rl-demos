#!/bin/bash
python pg_dist.py \
     --ps_hosts=localhost:2226\
     --worker_hosts=localhost:2227,localhost:2228\
     --job_name=worker --task_index=1
