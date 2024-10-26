
cd tools/torchscale-private
pip install -e .
cd ..
cd ..

python3 -m pip uninstall tutel -y
python3 -m pip install setuptools wheel
python3 -m pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@main

pip install git+https://github.com/yushuiwx/fairseq.git@moe3-v100
pip install git+https://github.com/shumingma/infinibatch.git
pip install iopath
pip install einops
pip install omegaconf tiktoken boto3