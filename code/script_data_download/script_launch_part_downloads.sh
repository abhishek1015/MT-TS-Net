#/bin/bash

for output in $(ls part_manifests_$1/*part[0-9])
do
	./gdc-client download -m $output -d /data/Jiang_Lab/Data/$1/ &
done
