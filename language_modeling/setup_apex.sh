
cd apex
pip3 install --user -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# pip install -v --no-cache-dir --global-option="--cpp_ext" \
#     --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" \
#     --global-option="--xentropy" --global-option="--fast_multihead_attn" \
#     git+git://github.com/NVIDIA/apex.git@e2083df5eb96643c61613b9df48dd4eea6b07690