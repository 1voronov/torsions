#!/bin/bash

num_replicas=50
for i in $(seq 1 $num_replicas); do
	working_dir_name="limit_cycle_$i"
	mkdir $working_dir_name

	dynamics_script=$(< dynamics.py)
	dynamics_script=$(echo "$dynamics_script" | sed "s/directory = \"limit_cycle\/\"/directory = \"$working_dir_name\/\"/")
	python -c "$dynamics_script" &> $working_dir_name/dynamics.log.$i &
done

wait

