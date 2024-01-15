# git clone https://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@github.com/msranlp/torchscale-moe.git -b shaohanh/flash-attn-v100
cd tools
cd torchscale-moe
pip install -e .
cd ..
cd ..
echo 'torchscale-moe ok'
pip install git+https://github.com/buaahsh/fairseq.git@moe
pip install git+https://github.com/shumingma/infinibatch.git
pip install iopath
pip install numpy==1.23.0
pip install tiktoken
pip install boto3
pip install sentencepiece
pip install scikit-learn
pip install datasets