# cp blob/connection.cfg /tmp/connection.cfg
# sudo mkdir /mnt1/msranlp

# sudo blobfuse /mnt1/msranlp/ --tmp-path=/mnt1/msranlp_blobfusetmp  --config-file=/tmp/connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other
# !!!!!make sure the ubuntu version is 20.04

wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install blobfuse -y

CN=xunwu
AN=msranlpintern
AK=TDiIfKsHfxN+/5CFjVoWvWT3s3feQWnbhmp8DH3coHZEZB1FRMIDaaC8ijOfLA+An4RB/fUNdpUXCjBgX64vdQ==

MOUNT_DIR=/mnt1/${AN}

CFG_PATH=~/fuse_connection_${CN}.cfg
MTP=/mnt1/localdata/blobfusetmp_temp_${CN}

sudo mkdir -p ${MTP}
sudo chmod 777 ${MTP}

printf 'accountName %s\naccountKey %s\ncontainerName %s\n' ${AN} ${AK} ${CN} > ${CFG_PATH}

sudo chmod 600 ${CFG_PATH}

sudo mkdir -p ${MOUNT_DIR}
sudo chmod 777 ${MOUNT_DIR}
blobfuse ${MOUNT_DIR} --tmp-path=${MTP}  --config-file=${CFG_PATH} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120