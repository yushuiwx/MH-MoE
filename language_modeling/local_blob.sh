wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
dpkg -i packages-microsoft-prod.deb
apt-get update
apt-get install blobfuse -y

CN=unilm
AN=msranlp
AK=OnSdaQJfGJ9YJ7tvFKNg9KkiQqXv1RHcc81oT5rxXmNzxWKTVF9vOmpn7+iIh44HoYG+h1pJFHNpIyrmwp2J0w==

MOUNT_DIR=/mnt/${AN}

CFG_PATH=~/fuse_connection_${CN}.cfg
MTP=/mnt/localdata/blobfusetmp_temp_${CN}

mkdir -p ${MTP}
chmod 777 ${MTP}

printf 'accountName %s\naccountKey %s\ncontainerName %s\n' ${AN} ${AK} ${CN} > ${CFG_PATH}

chmod 600 ${CFG_PATH}

mkdir -p ${MOUNT_DIR}
chmod 777 ${MOUNT_DIR}
blobfuse ${MOUNT_DIR} --tmp-path=${MTP}  --config-file=${CFG_PATH} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120


CN=unilm
AN=conversationhub
AK=Z9rLBJFxoJNDZWNK0MJ4ZXBFY6aKx11U8cyB19FfFhS/wZ8TlbY3vvQs21Kyq95e7cVtbvy5DZpzthEW2CcbzA==

MOUNT_DIR=/mnt/${AN}

CFG_PATH=~/fuse_connection_${CN}.cfg
MTP=/mnt/localdata/blobfusetmp_temp_${CN}

mkdir -p ${MTP}
chmod 777 ${MTP}

printf 'accountName %s\naccountKey %s\ncontainerName %s\n' ${AN} ${AK} ${CN} > ${CFG_PATH}

chmod 600 ${CFG_PATH}

mkdir -p ${MOUNT_DIR}
chmod 777 ${MOUNT_DIR}
blobfuse ${MOUNT_DIR} --tmp-path=${MTP}  --config-file=${CFG_PATH} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120



CN=tengchaolv
AN=conversationhubhot
AK=BH3e6RtvySZTVCPfFw5iqnzNSUwXMYtBMuBCF8tMa0dq++0wcJ7XdOYjlSasqExrdTqVi2l6rzS4+AStOIQsiQ==

MOUNT_DIR=/mnt/${AN}

CFG_PATH=~/fuse_connection_${CN}.cfg
MTP=/mnt/localdata/blobfusetmp_temp_${CN}

mkdir -p ${MTP}
chmod 777 ${MTP}

printf 'accountName %s\naccountKey %s\ncontainerName %s\n' ${AN} ${AK} ${CN} > ${CFG_PATH}

chmod 600 ${CFG_PATH}

mkdir -p ${MOUNT_DIR}
chmod 777 ${MOUNT_DIR}
blobfuse ${MOUNT_DIR} --tmp-path=${MTP}  --config-file=${CFG_PATH} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

CN=xunwu
AN=msranlpintern
AK=TDiIfKsHfxN+/5CFjVoWvWT3s3feQWnbhmp8DH3coHZEZB1FRMIDaaC8ijOfLA+An4RB/fUNdpUXCjBgX64vdQ==

MOUNT_DIR=/mnt/${AN}

CFG_PATH=~/fuse_connection_${CN}.cfg
MTP=/mnt/localdata/blobfusetmp_temp_${CN}

mkdir -p ${MTP}
chmod 777 ${MTP}

printf 'accountName %s\naccountKey %s\ncontainerName %s\n' ${AN} ${AK} ${CN} > ${CFG_PATH}

chmod 600 ${CFG_PATH}

mkdir -p ${MOUNT_DIR}
chmod 777 ${MOUNT_DIR}
blobfuse ${MOUNT_DIR} --tmp-path=${MTP}  --config-file=${CFG_PATH} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120


