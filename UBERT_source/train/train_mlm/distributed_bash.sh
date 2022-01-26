#!/bin/bash

`rm node* master* run_distributed* main_mlm.log mlm_INFO.log`
VAR1="node"
VAR2=".sh"
while getopts g:e:t:v: flag
do
    case "${flag}" in
        g) gpus=${OPTARG};;
        e) epochs=${OPTARG};;
        t) tbatch_size=${OPTARG};;
    esac
done

printf "#!/bin/bash\n\n#SBATCH --job-name=\"UMLS_ONLY distributed mlm training between ${gpus} gpus\"\n#SBATCH -D .\n#SBATCH --nodes=${gpus}\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:v100x:1\n#SBATCH --wait-all-nodes=1\n#SBATCH --mem=20g\n#SBATCH --time=2-00:00:00\n\n" > master${gpus}.sh

let upperlimit=$gpus-1;
for i in `eval echo {0..$upperlimit}`
do
    if [ $i == 0 ]
    then
        VAR3="$VAR1$i$VAR2"
        printf "#!/bin/bash\nhostname >> hostname.txt; source /data/$USER/conda/etc/profile.d/conda.sh; conda activate tft_clone; python /data/$USER/conda/envs/tft_clone/lib/python3.7/site-packages/torch/distributed/launch.py --nnode=$gpus --node_rank=$i --nproc_per_node=1 --master_addr=\`sed -n 1p hostname.txt\` --master_port=1234 /data/xxxx_UMLS_DL/xxxx/EXPERIMENTS/1_UMLS_ONLY/train_mlm/pretrain_umls_mlm.py --vocab_file=/data/xxxx_UMLS_DL/xxxx/EXPERIMENTS/tokenizers/umls_pubmed_pmc-vocab.txt --input_data=/data/xxxx_UMLS_DL/xxxx/EXPERIMENTS/datasets/umls_data/mrconso_atom_strings.txt --output_dir=./output_all --trained_model_dir=./trained_all --num_train_epochs=$epochs --per_device_train_batch_size=$tbatch_size" > $VAR3
        echo "srun --nodes=1 sh ${VAR3} &" >> master${gpus}.sh
    else
        VAR4="$VAR1$i$VAR2"
        printf "#!/bin/bash\nsource /data/$USER/conda/etc/profile.d/conda.sh; conda activate tft_clone; python /data/$USER/conda/envs/tft_clone/lib/python3.7/site-packages/torch/distributed/launch.py --nnode=$gpus --node_rank=$i --nproc_per_node=1 --master_addr=\`sed -n 1p hostname.txt\` --master_port=1234 /data/xxxx_UMLS_DL/xxxx/EXPERIMENTS/1_UMLS_ONLY/train_mlm/pretrain_umls_mlm.py --vocab_file=/data/xxxx_UMLS_DL/xxxx/EXPERIMENTS/tokenizers/umls_pubmed_pmc-vocab.txt --input_data=/data/xxxx_UMLS_DL/xxxx/EXPERIMENTS/datasets/umls_data/mrconso_atom_strings.txt --output_dir=./output_all --trained_model_dir=./trained_all --num_train_epochs=$epochs --per_device_train_batch_size=$tbatch_size" > $VAR4
        echo "srun --nodes=1 sh ${VAR4} &" >> master${gpus}.sh
    fi
done

`sed -i '$ s/.$//' master${gpus}.sh`

printf "rm hostname.txt\nsbatch master${gpus}.sh\n" > run_distributed_${gpus}.sh


