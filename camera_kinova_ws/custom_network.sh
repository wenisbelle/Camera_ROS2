#!/bin/bash

PARENT_INTERFACE="enp0s31f6"

if docker network inspect macvlan_net > /dev/null 2>&1; then
    echo "Docker network 'macvlan_net' already exists"
else
    echo "Creating the 'macvlan_net' network"
    docker network create \
        -d macvlan \
        --subnet=192.168.1.0/24 \
        --gateway=192.168.1.1 \
        --ip-range=192.168.1.96/28 \
        -o parent=$PARENT_INTERFACE \
        macvlan_net
fi

    

