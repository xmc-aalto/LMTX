#!/bin/bash

dataset=$1
MAX_EPOCHES=$2
batch_size_llm=$3
prompt_id=$4
num_train_instances=$5
best_p1=0.0
best_model_epoch=0

shuf -n 800 $dataset/trn.json -o  $dataset/dev.json
shuf -n $num_train_instances $dataset/trn.json -o  $dataset/trn_shuf.json
start_time=$(date +%s)
echo "start time: $start_time"
for epoch in `seq 0 1 $MAX_EPOCHES`; 
do
	if [ $epoch == 0 ]
	then
  		python src/get_shortlist.py --share_weight \
		--labels_path $dataset/lbl.json \
		--model_name 'sentence-transformers/msmarco-distilbert-base-v4' \
		--log_dir log_dir --eval_batch_size 640 --val_path $dataset/trn_shuf.json \
		--max_seq_length 430 --max_label_length 15 \
		--output_shortlist_path $dataset/trn_shortlist_$epoch.json --encode_title
	else
		model_name_epoch=$((epoch-1))
		python src/get_shortlist.py --share_weight \
		--labels_path $dataset/lbl.json \
		--model_name 'sentence-transformers/msmarco-distilbert-base-v4' \
		--log_dir log_dir --eval_batch_size 640 --val_path $dataset/trn_shuf.json \
		--max_seq_length 430 --max_label_length 15 \
		--output_shortlist_path $dataset/trn_shortlist_$epoch.json \
		--model_path $dataset/model_dir/biencoder_iter_$model_name_epoch.pt --encode_title
	fi

	echo "epoch {$epoch} shortlist finished"

	python src/inference_wizardlm_batch.py \
	--trn_docs_path $dataset/trn_shuf.json \
	--model_path wizardlm13b --shortlist_path $dataset/trn_shortlist_$epoch.json \
	--ouput_trn_llm_path $dataset/trn_pseudo_$epoch.json --labels_path $dataset/lbl.json \
	--topk 10 --batch_size $batch_size_llm --max_seq_length 430 \
	--prompt_id $prompt_id --encode_title

	echo "epoch {$epoch} llm feedback finished"
	
	python src/train_biencoder.py \
		--trn_docs_path $dataset/trn_pseudo_$epoch.json \
		--share_weight --labels_path $dataset/lbl.json \
		--model_name 'sentence-transformers/msmarco-distilbert-base-v4' \
		--output_dir $dataset/model_dir --train_batch_size 128 --num_epoches 1 \
		--val_path $dataset/tst.json --max_seq_length 450 --max_label_length 15 \
		--validate_interval 1 --seed 108 --loss_agressive --eval_batch_size 512 \
		--iteration_epoch $epoch --encode_title
	echo "epoch {$epoch} bi-encoder training finished"

	# evaluate the model
	python src/get_shortlist.py --share_weight \
		--labels_path $dataset/lbl.json \
		--model_name 'sentence-transformers/msmarco-distilbert-base-v4' \
		--log_dir log_dir --eval_batch_size 640 --val_path $dataset/dev.json \
		--max_seq_length 430 --max_label_length 15 \
		--output_shortlist_path $dataset/dev_shortlist.json \
		--model_path $dataset/model_dir/biencoder_iter_$epoch.pt --encode_title

	python src/inference_wizardlm_batch.py \
	--trn_docs_path $dataset/dev.json \
	--model_path wizardlm13b --shortlist_path $dataset/dev_shortlist.json \
	--ouput_trn_llm_path $dataset/dev_pseudo.json --labels_path $dataset/lbl.json \
	--topk 2 --batch_size $batch_size_llm --max_seq_length 430 --prompt_id $prompt_id --encode_title

	python src/evaluate_with_llm.py --pseudo_trn_path $dataset/dev_pseudo.json \
	--shortlist_path $dataset/dev_shortlist.json \
	--p1_output_path $dataset/model_dir/current_p1.txt

	p1_file=$dataset/model_dir/current_p1.txt
	p1=$(cat $p1_file)
	echo "epoch {$epoch} current p@1 based on llm: $p1"
	if [ `echo "$p1 > $best_p1" | bc` -eq 1 ]
	then
		best_p1=$p1	
		best_model_epoch=$epoch
	else
		break
	fi
done  
echo "best model: biencoder_iter_$best_model_epoch.pt"
end_time=$(date +%s)
echo "end time: $end_time"
diff_time=$(( $end_time - $start_time ))
echo "Training took $diff_time seconds"
