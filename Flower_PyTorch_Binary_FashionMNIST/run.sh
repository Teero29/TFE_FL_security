#!/bin/bash

echo "Starting server"
python server.py &
sleep 3  

for i in `seq 0 1`; do
    echo "Starting client $i"
    python client.py &
done


trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
