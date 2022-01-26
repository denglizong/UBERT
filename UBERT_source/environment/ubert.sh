source /data/$USER/conda/etc/profile.d/conda.sh
conda create -n ubert python=3.7 -y
conda activate ubert
conda install cudatoolkit=10.0 -y
conda install cudnn=7.3.1 -y
yes|pip3 install tensorflow-gpu==1.15
yes|pip3 install torch==1.8.1 transformers==4.5 
yes|pip3 install -U scikit-learn
yes|pip install fairscale
