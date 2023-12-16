for seed in 123456; do
  CUDA_VISIBLE_DEVICES=1 python fairseq/fairseq_cli/train.py fairseq/data-bin/multi30k_en_de  \
  --arch MMTFA2I \
  --task MultimodalT \
  --share-all-embeddings \
  --clip-norm 0 \
  --optimizer adam \
  --reset-optimizer \
  --lr 0.005 \
  --source-lang en \
  --target-lang de \
  --max-tokens 4096 \
  --update-freq 4 \
  --weight-decay 0.1 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.2 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 2000 \
  --warmup-init-lr 1e-07 \
  --adam-betas 0.9,0.98 \
  --dropout 0.3 \
  --save-dir  model/MMT \
  --eval-bleu \
  --seed $seed \
  --keep-best-checkpoints 3 \
  --best-checkpoint-metric bleu \
  --eval-bleu-detok moses \
  --maximize-best-checkpoint-metric \
  --eval-bleu-remove-bpe \
  --keep-last-epochs 10 \
  --max-update 8000 \
  --patience 15 \

  CUDA_VISIBLE_DEVICES=1 python3 scripts/average_checkpoints.py --inputs fairseq/model/MMT --output fairseq/model/MMTSaved/${seed}.ensemble.pt --num-epoch-checkpoints 10

  echo "The start of this combination" >> Results.txt
  for dataset in test2016 test2017 test2017mscoco; do
      outputBleu=$(CUDA_VISIBLE_DEVICES=1 python fairseq/fairseq_cli/generate.py \
      fairseq/data-bin/multi30k_en_de/$dataset --task MultimodalT --path fairseq/model/MMTSaved/${seed}.ensemble.pt  --beam 5 --lenpen 1 --batch-size 128 --remove-bpe | grep "Generate test with beam=")

      if [[ ! -z $outputBleu ]]; then
  echo "dataset"=$dataset, "t=$seed, $outputBleu" >> Results.txt
      fi
      outputMeteor=$(CUDA_VISIBLE_DEVICES=1 python fairseq/fairseq_cli/generate.py \
     fairseq/data-bin/multi30k_en_de/$dataset --task MultimodalT --scoring meteor --path fairseq/model/MMTSaved/${seed}.ensemble.pt  --beam 5 --lenpen 1 --batch-size 128 --remove-bpe | grep "Generate test with beam=")
      if [[ ! -z $outputMeteor ]]; then
  echo "dataset"=$dataset, "t=$seed, $outputMeteor" >> Results.txt
      fi
  done
  echo "The end of this combination" >>Results.txt
  echo "" >> Results.txt
  rm -rf fairseq/model/MMT/*
      done
done
