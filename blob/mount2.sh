cp blob/connection.cfg /tmp/connection.cfg
sudo mkdir /mnt/msranlp

sudo blobfuse /mnt/msranlp/ --tmp-path=/mnt/msranlp_blobfusetmp  --config-file=/tmp/connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other