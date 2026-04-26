#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collator + CustomTrainer (SFTベース)
"""

import os

import numpy as np
import torch
import editdistance
from PIL import Image as PILImage
from transformers import (
    Trainer,
    DefaultDataCollator,
)
from torch.utils.tensorboard import SummaryWriter


class Collator(DefaultDataCollator):
    def __init__(self, processor):
        super().__init__()
        self.tokenizer = processor.tokenizer

    def __call__(self, batch):
        pixel_values = [b.pop('pixel_values') for b in batch]
        image_grid_thw = [b.pop('image_grid_thw') for b in batch]
        surrounding_texts = [b.pop('surrounding_texts', []) for b in batch]
        masked_image_paths = [b.pop('masked_image_path') for b in batch]

        # 画像サイズ可変により sequence 長が異なるため、バッチ内最大長にパディング
        max_len = max(b['input_ids'].shape[0] for b in batch)
        pad_id = self.tokenizer.pad_token_id or 0
        for b in batch:
            pad_len = max_len - b['input_ids'].shape[0]
            if pad_len > 0:
                b['input_ids'] = torch.cat([
                    b['input_ids'], torch.full((pad_len,), pad_id, dtype=b['input_ids'].dtype)
                ])
                b['attention_mask'] = torch.cat([
                    b['attention_mask'], torch.zeros(pad_len, dtype=b['attention_mask'].dtype)
                ])
                if 'labels' in b:
                    b['labels'] = torch.cat([
                        b['labels'], torch.full((pad_len,), -100, dtype=b['labels'].dtype)
                    ])
                if 'mm_token_type_ids' in b:
                    b['mm_token_type_ids'] = torch.cat([
                        b['mm_token_type_ids'], torch.zeros(pad_len, dtype=b['mm_token_type_ids'].dtype)
                    ])

        processed_batch = super().__call__(batch)
        processed_batch['pixel_values'] = torch.cat(pixel_values, dim=0)
        processed_batch['image_grid_thw'] = torch.cat(image_grid_thw, dim=0)
        processed_batch['masked_image_paths'] = masked_image_paths
        processed_batch['surrounding_texts'] = surrounding_texts
        return processed_batch


class CustomTrainer(Trainer):
    """base_loss (cross entropy) のみで学習するカスタムTrainer"""

    def __init__(
        self,
        *args,
        compute_metrics_freq=100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_metrics_freq = compute_metrics_freq
        self.tb_writer = None

    def _compute_surrounding_penalty(self, logits, labels, surrounding_texts_batch):
        """周辺テキストと一致するトークンを予測した場合のunlikelihood loss"""
        total_penalty = torch.tensor(0.0, device=logits.device)
        valid_count = 0

        for i, surr_texts in enumerate(surrounding_texts_batch):
            if not surr_texts:
                continue
            resp_mask = labels[i, 1:] != -100
            if resp_mask.sum() == 0:
                continue

            gt_tokens = set(labels[i, 1:][resp_mask].tolist())

            surr_token_ids = set()
            for text in surr_texts[:3]:
                ids = self.processing_class.tokenizer.encode(text, add_special_tokens=False)
                surr_token_ids.update(ids)

            penalty_ids = list(surr_token_ids - gt_tokens)
            if not penalty_ids:
                continue

            resp_logits = logits[i, :-1][resp_mask]
            probs = torch.softmax(resp_logits, dim=-1)
            penalty_tensor = torch.tensor(penalty_ids, device=logits.device, dtype=torch.long)
            penalty_prob = probs[:, penalty_tensor].sum(dim=-1).clamp(max=1 - 1e-7)
            total_penalty = total_penalty + (-torch.log(1 - penalty_prob)).mean()
            valid_count += 1

        return total_penalty / valid_count if valid_count > 0 else total_penalty

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        masked_image_paths = inputs.pop('masked_image_paths', None)
        surrounding_texts_batch = inputs.pop('surrounding_texts', None)
        outputs = model(**inputs)
        loss = outputs.loss

        if not (self.model.training and labels is not None):
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits

        if surrounding_texts_batch:
            penalty = self._compute_surrounding_penalty(logits, labels, surrounding_texts_batch)
            loss = loss + 0.1 * penalty
        batch_size = labels.shape[0]
        current_step = self.state.global_step if self.state else 0

        # 監視メトリクス（100ステップごとのみ・勾配不要）
        if current_step % self.compute_metrics_freq == 0:
            with torch.no_grad():
                predicted_ids = torch.argmax(logits, dim=-1)
                total_accuracy = 0.0
                total_cer = 0.0
                total_ned = 0.0
                total_anls = 0.0
                mon_valid = 0

                for i in range(batch_size):
                    # logits[t] predicts position t+1, so align with labels[t+1]
                    resp_mask = labels[i, 1:] != -100
                    if resp_mask.sum() == 0:
                        continue
                    mon_valid += 1
                    pred_tokens = predicted_ids[i, :-1][resp_mask]
                    label_tokens = labels[i, 1:][resp_mask]

                    total_accuracy += (pred_tokens == label_tokens).float().mean().item()

                    pred_text = self.processing_class.tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                    label_text = self.processing_class.tokenizer.decode(label_tokens, skip_special_tokens=True).strip()
                    total_cer += editdistance.eval(pred_text, label_text) / max(len(label_text), 1)

                    ed = editdistance.eval(pred_text, label_text)
                    max_len = max(len(pred_text), len(label_text), 1)
                    ned = ed / max_len
                    total_ned += ned
                    total_anls += 1.0 - ned

                if mon_valid > 0:
                    self.log(
                        {
                            'train/accuracy': total_accuracy / mon_valid,
                            'train/cer': total_cer / mon_valid,
                            'train/ned': total_ned / mon_valid,
                            'train/anls': total_anls / mon_valid,
                            'train/base_loss': loss.item(),
                        }
                    )

                if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                    self._log_predictions(current_step, predicted_ids, labels, masked_image_paths)

        return (loss, outputs) if return_outputs else loss

    def _log_predictions(self, step, predicted_ids, labels, masked_image_paths=None):
        """学習中の予測結果をテキスト＋対応masked画像としてTensorBoardにログ"""
        if self.tb_writer is None:
            log_dir = os.environ.get('TENSORBOARD_LOGGING_DIR', self.args.logging_dir)
            self.tb_writer = SummaryWriter(log_dir=log_dir)

        sample_idx = next(
            (i for i in range(labels.shape[0]) if (labels[i] != -100).sum() > 0),
            None,
        )
        if sample_idx is None:
            return
        # logits[t] predicts position t+1, so align with labels[t+1]
        sample_mask = labels[sample_idx, 1:] != -100
        pred_tokens = predicted_ids[sample_idx, :-1][sample_mask]
        predicted_text = self.processing_class.tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
        ground_truth_text = self.processing_class.tokenizer.decode(
            labels[sample_idx, 1:][sample_mask], skip_special_tokens=True
        ).strip()

        ed = editdistance.eval(predicted_text, ground_truth_text)
        cer = ed / max(len(ground_truth_text), 1)
        max_len = max(len(predicted_text), len(ground_truth_text), 1)
        ned = ed / max_len
        anls = 1.0 - ned

        text_log = (
            f'**Predicted:** {predicted_text}\n\n'
            f'**Ground Truth:** {ground_truth_text}\n\n'
            f'CER: `{cer:.4f}` | NED: `{ned:.4f}` | ANLS: `{anls:.4f}`'
        )
        self.tb_writer.add_text('predictions/greedy', text_log, step)

        if masked_image_paths and masked_image_paths[0] and os.path.exists(masked_image_paths[0]):
            try:
                masked_image = PILImage.open(masked_image_paths[0]).convert('RGB')
                self.tb_writer.add_image(
                    'predictions/masked_image',
                    np.array(masked_image).transpose(2, 0, 1),
                    step,
                )
            except Exception:
                pass
