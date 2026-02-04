cd moshi-finetune                                                                
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node 2 -m train example/stt_zh_lora_local.yaml 