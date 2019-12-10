#!/usr/bin/env bash
python test.py --data_dir ./sample_data/test_data.txt --ngf 64 --batch_size 1 --model test --name A2I --norm batch --gpu_ids 0 --eval --verbose