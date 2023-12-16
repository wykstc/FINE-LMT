CUDA_VISIBLE_DEVICES=0 python fairseq/fairseq_cli/generate.py \
fairseq/data-bin/multi30k_en_de/test2016 --task MultimodalT --path Results.txtfairseq/model/MMTSaved/123456.ensemble.pt --beam 5 --lenpen 1 --batch-size 128 --remove-bpe


CUDA_VISIBLE_DEVICES=0 python fairseq/fairseq_cli/generate.py \
fairseq/data-bin/multi30k_en_de/test2016 --task MultimodalT --scoring meteor --path Results.txtfairseq/model/MMTSaved/123456.ensemble.pt --beam 5 --lenpen 1 --batch-size 128 --remove-bpe
