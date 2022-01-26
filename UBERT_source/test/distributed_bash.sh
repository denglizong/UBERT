#!/bin/bash

`rm node* master* run_distributed* main_method.log transformer*`
VAR1="node"
VAR2=".sh"
while getopts g:v:c: flag
do
    case "${flag}" in
        g) gpus=${OPTARG};;
        v) ebatch_size=${OPTARG};;
        c) checkpoint_name=${OPTARG};;
    esac
done

printf "#!/bin/bash\n\n#SBATCH --job-name=\"testing distributed between ${gpus} gpus\"\n#SBATCH -D .\n#SBATCH --nodes=${gpus}\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:v100x:1\n#SBATCH --wait-all-nodes=1\n#SBATCH --mem=50g\n#SBATCH --time=15:00:00\n\n" > master${gpus}.sh

let upperlimit=$gpus-1;
for i in `eval echo {0..$upperlimit}`
do
    if [ $i == 0 ]
    then
        VAR3="$VAR1$i$VAR2"
        printf "#!/bin/bash\nhostname >> hostname.txt; source /data/$USER/conda/etc/profile.d/conda.sh; conda activate tft; python /data/$USER/conda/envs/tft/lib/python3.7/site-packages/torch/distributed/launch.py --nnode=$gpus --node_rank=$i --nproc_per_node=1 --master_addr=\`sed -n 1p hostname.txt\` --master_port=1234 /data/xxxx/xxxx/EXPERIMENTS/1_UMLS_ONLY/testing/testing_sp_optimized_w_sharded_ddp.py --vocab_file=/data/xxxx/xxxx/EXPERIMENTS/tokenizers/umls_pubmed_pmc-vocab.txt --aui_vec_file_path=/data/xxxx/xxxx/EXPERIMENTS/aui_vec/aui_vec.pkl --test_data=/data/xxxx/xxxx/EXPERIMENTS/datasets/2020AA-ACTIVE_ALL_MODEL_GENTEST_DS_TEST_DS_SHUF.RRF --out_dir=out_test_all --per_device_eval_batch_size=$ebatch_size --checkpoint_model=/data/xxxx/xxxx/EXPERIMENTS/1_UMLS_ONLY/train_sp/out_all_correct_metric/$checkpoint_name --main_method_log=./correct_metrics_main_method_${checkpoint_name}.log --checkpoint_name=$checkpoint_name"> $VAR3
        echo "srun --nodes=1 sh ${VAR3} &" >> master${gpus}.sh
    else
        VAR4="$VAR1$i$VAR2"
        printf "#!/bin/bash\nsource /data/$USER/conda/etc/profile.d/conda.sh; conda activate tft; python /data/$USER/conda/envs/tft/lib/python3.7/site-packages/torch/distributed/launch.py --nnode=$gpus --node_rank=$i --nproc_per_node=1 --master_addr=\`sed -n 1p hostname.txt\` --master_port=1234 /data/xxxx/xxxx/EXPERIMENTS/1_UMLS_ONLY/testing/testing_sp_optimized_w_sharded_ddp.py --vocab_file=/data/xxxx/xxxx/EXPERIMENTS/tokenizers/umls_pubmed_pmc-vocab.txt --aui_vec_file_path=/data/xxxx/xxxx/EXPERIMENTS/aui_vec/aui_vec.pkl --test_data=/data/xxxx/xxxx/EXPERIMENTS/datasets/2020AA-ACTIVE_ALL_MODEL_GENTEST_DS_TEST_DS_SHUF.RRF --out_dir=out_test_all --per_device_eval_batch_size=$ebatch_size --checkpoint_model=/data/xxxx/xxxx/EXPERIMENTS/1_UMLS_ONLY/train_sp/out_all_correct_metric/$checkpoint_name --main_method_log=./correct_metrics_main_method_${checkpoint_name}.log --checkpoint_name=$checkpoint_name"> $VAR4
        echo "srun --nodes=1 sh ${VAR4} &" >> master${gpus}.sh
    fi
done

`sed -i '$ s/.$//' master${gpus}.sh`

printf "rm hostname.txt\nsbatch master${gpus}.sh\n" > run_distributed_${gpus}.sh


