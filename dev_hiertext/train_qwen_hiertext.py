#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HierTextデータセットを使用した学習スクリプト
マスク予測メイン + 座標補助のアプローチ
"""

import os
import logging
import argparse

import torch
from peft import TaskType, PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForImageTextToText,
    trainer_utils,
)

from dpo_trainer import DPOTrainer
from custom_trainer import Collator, CustomTrainer
from hiertext_dataset import Dataset

logging.getLogger('torch.utils.checkpoint').setLevel(logging.ERROR)


# =====================
# 学習処理
# =====================
def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='HierText dataset training script')
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help=(
            'Path to checkpoint to resume training from, or "True" to auto-detect latest checkpoint '
            '(e.g., ./qwen_model_checkpoints_hiertext/checkpoint-18000 or True)'
        ),
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None,
        help='Maximum number of training samples to use (for debugging)',
    )
    parser.add_argument(
        '--max_val_samples',
        type=int,
        default=200,
        help='Maximum number of validation samples to use',
    )
    parser.add_argument(
        '--paraphrase_cache',
        type=str,
        default='./paraphrase_cache_train.json',
        help='パラフレーズキャッシュJSONファイルのパス',
    )
    parser.add_argument(
        '--dpo',
        action='store_true',
        help='SFT完了後にDPOフェーズを実行する',
    )
    args = parser.parse_args()

    # resume_from_checkpointの処理: "True"という文字列をbooleanに変換
    if args.resume_from_checkpoint and args.resume_from_checkpoint.lower() == 'true':
        args.resume_from_checkpoint = True

    os.environ['TENSORBOARD_LOGGING_DIR'] = './logs_v3'

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device_name = f'cuda:{local_rank}'

    model_id = 'Qwen/Qwen3-VL-2B-Instruct'

    # QLoRA設定
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id, padding_side='left')

    # HierTextデータセット（annotationのみロード、VRAMゼロ）
    print('Loading HierText train dataset...')
    train_dataset = Dataset(
        annotation_file='/home/user/dev/dev_hiertext/hiertext/gt/train.jsonl.gz',
        processor=processor,
        max_samples=args.max_train_samples,
        mask_dir='/home/user/dev/dev_hiertext/hiertext/Mask_Monochro_train',
        masked_image_dir='/home/user/dev/dev_hiertext/hiertext/Masked_Images_train',
        augment=True,
    )

    # パラフレーズキャッシュのロード（generate_paraphrases.py で事前生成したファイルを読み込む）
    if os.path.exists(args.paraphrase_cache):
        train_dataset.load_paraphrase_cache(args.paraphrase_cache)
    else:
        print(f'Paraphrase cache not found at {args.paraphrase_cache}, skipping.')

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        attn_implementation='eager',
        device_map={'': device_name},
    )
    model = prepare_model_for_kbit_training(model)

    # チェックポイント検出（モデル初期化時に必要）
    checkpoint_path = None
    if args.resume_from_checkpoint:
        if isinstance(args.resume_from_checkpoint, bool) or args.resume_from_checkpoint.lower() == 'true':
            checkpoint_path = trainer_utils.get_last_checkpoint('./qwen_model_checkpoints_hiertext_v3')
        else:
            checkpoint_path = args.resume_from_checkpoint

    # LoRA設定とアダプターのロード
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'Loading LoRA adapter from checkpoint: {checkpoint_path}')
        # アダプターは lora_sft サブディレクトリに保存されている
        adapter_path = os.path.join(checkpoint_path, 'lora_sft')
        if os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name='lora_sft')
        else:
            # 古い形式（直接保存）の場合
            model = PeftModel.from_pretrained(model, checkpoint_path, adapter_name='lora_sft')
        model.set_adapter('lora_sft')
    else:
        print('Initializing new LoRA adapter')
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=['q_proj', 'v_proj'],
            bias='none',
        )
        model = get_peft_model(model, lora_config, adapter_name='lora_sft')
        model.set_adapter('lora_sft')

    print('Saving augmented sample images...')
    train_dataset.save_augmented_samples(n=10, output_dir='./aug_samples')

    print('Loading HierText validation dataset...')
    val_dataset = Dataset(
        annotation_file='/home/user/dev/dev_hiertext/hiertext/gt/validation.jsonl.gz',
        processor=processor,
        max_samples=args.max_val_samples,
        mask_dir='/home/user/dev/dev_hiertext/hiertext/Mask_Monochro_val',
        masked_image_dir='/home/user/dev/dev_hiertext/hiertext/Masked_Images_val',
        augment=False,
    )

    collator = Collator(processor)

    # TrainingArguments設定
    training_args = TrainingArguments(
        output_dir='./qwen_model_checkpoints_hiertext_v3',
        per_device_train_batch_size=4,
        num_train_epochs=6,
        learning_rate=2e-5,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=['tensorboard'],
        optim='paged_adamw_8bit',
        ddp_find_unused_parameters=False,
        label_smoothing_factor=0.1,
    )

    # Trainer設定
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor,
        length_penalty_weight=0.1,
        surrounding_context_loss_weight=0.05,
        embedding_model_id='Qwen/Qwen3-VL-Embedding-2B',
        compute_metrics_freq=100,
        inference_log_freq=100,
        num_viz_samples=3,
    )

    # 学習実行
    print('Starting training with HierText dataset...')

    # チェックポイントから学習を再開（オプティマイザー・スケジューラーの状態を復元）
    # モデルの重みは既にロード済み
    if checkpoint_path:
        print(f'Resuming training from checkpoint: {checkpoint_path}')
        # Trainerのモデルロード処理を無効化（LoRAアダプターは既にロード済み）
        trainer._load_from_checkpoint = lambda resume_from_checkpoint: None
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print('Starting training from scratch.')
        trainer.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================
    # DPOフェーズ（SFT完了後）
    # =====================
    if not args.dpo:
        print('DPO phase skipped. Use --dpo to enable.')
        return

    print('\nStarting DPO fine-tuning phase...')
    dpo_args = TrainingArguments(
        output_dir='./qwen_model_checkpoints_hiertext_v3_dpo',
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        logging_steps=100,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=['tensorboard'],
        logging_dir='./logs_v3_dpo',
        optim='paged_adamw_8bit',
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor,
        length_penalty_weight=0.0,
        vision_alignment_loss_weight=0.0,
        compute_metrics_freq=100,
        inference_log_freq=100,
        num_viz_samples=3,
        dpo_weight=0.1,
        dpo_beta=0.1,
        dpo_num_candidates=3,
        dpo_num_samples=2,
    )
    dpo_trainer.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('\nSaving model and tokenizer to ./qwen_model_hiertext_v3')
    model.save_pretrained('./qwen_model_hiertext_v3')
    processor.save_pretrained('./qwen_model_hiertext_v3')
    print('Model saved successfully!')
    print('HierText training completed.')


if __name__ == '__main__':
    main()
