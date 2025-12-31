set -v
datasets=("NELL-995-subset-inductive")
# datasets=("WN18RR-subset" "NELL-995-subset-inductive")
suffixs=("_1000")
# suffixs=("_full" "_2000" "_1000")
# suffix="_1000"
finding_modes=("head")
# finding_modes=("head" "tail")
finding_mode="tail"
device="cuda:0"
seed=42
relation_prediction_lr=1e-5


# "relation prediction"
for dataset in ${datasets[@]}; do
  for suffix in ${suffixs[@]}; do
    for finding_mode in ${finding_modes[@]}; do
      python relation_prediction.py --device $device --epochs 3 --batch_size 1 --dataset $dataset --max_path_num 6 --learning_rate $relation_prediction_lr --neg_sample_num_train 5 --neg_sample_num_valid 5 --neg_sample_num_test 50 --mode $finding_mode --seed $seed --suffix $suffix --do_train --do_test
    done
  done
done
