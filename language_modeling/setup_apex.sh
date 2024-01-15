# git clone https://github.com/ptrblck/apex.git
cd apex
# git checkout apex_no_distributed
# pip install -v --no-cache-dir ./
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip3 install --user -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..