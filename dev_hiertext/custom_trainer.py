#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collator + CustomTrainer (SFTベース)
"""

import os

import numpy as np
import torch
import editdistance
import torch.nn.functional as F
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
        surrounding_texts = [b.pop('surrounding_texts') for b in batch]
        masked_image_paths = [b.pop('masked_image_path') for b in batch]
        processed_batch = super().__call__(batch)
        processed_batch['pixel_values'] = torch.cat(pixel_values, dim=0)
        processed_batch['image_grid_thw'] = torch.cat(image_grid_thw, dim=0)
        processed_batch['surrounding_texts'] = surrounding_texts
        processed_batch['masked_image_paths'] = masked_image_paths
        return processed_batch


class CustomTrainer(Trainer):
    """トークン数乖離ペナルティ + Qwen3-VL-Embedding周辺コンテキスト整合性lossを適用するカスタムTrainer"""

    def __init__(
        self,
        *args,
        length_penalty_weight=0.1,
        surrounding_context_loss_weight=0.05,
        embedding_model_id='Qwen/Qwen3-VL-Embedding-8B',
        compute_metrics_freq=100,
        inference_log_freq=100,
        num_viz_samples=3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.length_penalty_weight = length_penalty_weight
        self.surrounding_context_loss_weight = surrounding_context_loss_weight
        self._embedding_model_id = embedding_model_id
        self.compute_metrics_freq = compute_metrics_freq
        self.inference_log_freq = inference_log_freq
        self.num_viz_samples = num_viz_samples
        self.global_step_count = 0
        self.tb_writer = None
        self._embed_model = None
        self._embed_proc = None

    def _get_embed_model(self, device):
        """Qwen3-VL-Embedding-8B を遅延ロード（bfloat16、eval固定）"""
        if self._embed_model is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self._embed_proc = AutoProcessor.from_pretrained(
                self._embedding_model_id, use_fast=False, trust_remote_code=True
            )
            self._embed_model = AutoModelForImageTextToText.from_pretrained(
                self._embedding_model_id,
                dtype=torch.bfloat16,
                device_map={'': device},
                trust_remote_code=True,
            )
            self._embed_model.eval()
        return self._embed_model, self._embed_proc

    def _get_embedding(self, embed_model, enc_inputs):
        """embed_model の forward から最終層 hidden state を取得して平均プーリング"""
        out = embed_model(**enc_inputs, output_hidden_states=True)
        # hidden_states: tuple of [batch, seq_len, dim], last layer
        last_hidden = out.hidden_states[-1]  # [batch, seq_len, dim]
        return last_hidden[:, -1, :]  # last token pooling

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """カスタム損失関数（微分可能な期待長さペナルティ + Qwen3-VL-Embedding周辺コンテキスト整合性loss）"""
        labels = inputs.get('labels')
        surrounding_texts = inputs.pop('surrounding_texts', None)
        masked_image_paths = inputs.pop('masked_image_paths', None)
        outputs = model(**inputs)
        base_loss = outputs.loss

        if not (self.model.training and labels is not None):
            return (base_loss, outputs) if return_outputs else base_loss

        logits = outputs.logits
        batch_size = labels.shape[0]
        current_step = self.state.global_step if self.state else 0

        # EOS直接予測loss: 応答終端の1つ前のlogitsでEOSをCE lossで強制予測
        eos_id = self.processing_class.tokenizer.convert_tokens_to_ids('<|im_end|>')

        total_length_loss = logits.new_zeros(())
        valid_samples = 0

        for i in range(batch_size):
            resp_positions = labels[i].ne(-100).nonzero(as_tuple=False).squeeze(-1)
            if len(resp_positions) < 2:
                continue
            valid_samples += 1

            # logits[t] は labels[t+1] を予測するため、最後のラベル（EOS）は
            # resp_positions[-1] のラベルであり、それを予測するのは resp_positions[-2] のlogits
            eos_pred_pos = resp_positions[-2]
            eos_logit = logits[i][eos_pred_pos].unsqueeze(0)
            eos_target = torch.tensor([eos_id], device=logits.device)
            total_length_loss = total_length_loss + F.cross_entropy(eos_logit, eos_target)

        if valid_samples > 0:
            avg_length_loss = total_length_loss / valid_samples
            loss = base_loss + self.length_penalty_weight * avg_length_loss
        else:
            avg_length_loss = logits.new_zeros(())
            loss = base_loss

        # Qwen3-VL-Embedding による周辺コンテキスト整合性 loss:
        #   ① softmax(logits) @ embed_table で soft embedding を生成（微分可能）
        #   ② 周辺画像・周辺テキストを Qwen3-VL-Embedding でエンコード（no_grad）
        #   ③ 同一埋め込み空間でコサイン距離を最小化 → logits まで勾配が届く
        avg_surrounding_ctx_loss = logits.new_zeros(())
        valid_ctx = 0
        if (self.model.training
                and self.surrounding_context_loss_weight > 0
                and surrounding_texts
                and masked_image_paths):
            device = logits.device
            embed_model, embed_proc = self._get_embed_model(device)
            _img_token = getattr(embed_proc, 'image_token', '<|image_pad|>')

            # embed_table は凍結済み（パラメータ更新不要、activationのみ追跡）
            embed_table = embed_model.get_input_embeddings().weight  # [vocab_embed, hidden_embed]

            ctx_loss_accum = logits.new_zeros(())

            for i in range(batch_size):
                resp_mask = labels[i] != -100
                if resp_mask.sum() == 0 or not surrounding_texts[i]:
                    continue
                if not (masked_image_paths[i] and os.path.exists(masked_image_paths[i])):
                    continue

                # ① soft embedding: softmax(logits) @ embed_table（勾配あり）
                probs = torch.softmax(logits[i][resp_mask].float(), dim=-1)  # [seq_len, vocab_vl]
                vocab_size = min(probs.shape[-1], embed_table.shape[0])
                soft_input_embeds = probs[:, :vocab_size] @ embed_table[:vocab_size].float()  # [seq_len, hidden_embed]

                seq_len = soft_input_embeds.shape[0]
                attn_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
                out = embed_model(
                    inputs_embeds=soft_input_embeds.unsqueeze(0).to(embed_table.dtype),
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                )
                pred_embed = F.normalize(out.hidden_states[-1][0, -1, :].float(), dim=-1)  # [dim]

                # ② 周辺画像・周辺テキストは勾配不要
                with torch.no_grad():
                    img = PILImage.open(masked_image_paths[i]).convert('RGB')
                    img_enc = embed_proc(
                        text=[_img_token], images=[img], return_tensors='pt'
                    ).to(device)
                    img_embed = F.normalize(
                        self._get_embedding(embed_model, img_enc).squeeze(0).float(), dim=-1
                    )  # [dim]

                    ctx_texts = surrounding_texts[i][:8]
                    ctx_enc = embed_proc(
                        text=ctx_texts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=64,
                    ).to(device)
                    ctx_text_embed = F.normalize(
                        self._get_embedding(embed_model, ctx_enc).float().mean(dim=0), dim=-1
                    )  # [dim]

                    context_embed = F.normalize(img_embed + ctx_text_embed, dim=-1)  # [dim]

                ctx_loss = torch.clamp(1.0 - (pred_embed * context_embed).sum(), max=1.0)
                ctx_loss_accum = ctx_loss_accum + ctx_loss
                valid_ctx += 1

            if valid_ctx > 0:
                avg_surrounding_ctx_loss = ctx_loss_accum / valid_ctx
                loss = loss + self.surrounding_context_loss_weight * avg_surrounding_ctx_loss

        # 監視メトリクス（100ステップごとのみ・勾配不要）
        if current_step % self.compute_metrics_freq == 0:
            with torch.no_grad():
                predicted_ids = torch.argmax(logits, dim=-1)
                total_accuracy = 0.0
                total_cer = 0.0
                mon_valid = 0

                total_ned = 0.0
                total_anls = 0.0

                for i in range(batch_size):
                    resp_mask = labels[i] != -100
                    if resp_mask.sum() == 0:
                        continue
                    mon_valid += 1
                    pred_tokens = predicted_ids[i][resp_mask]
                    label_tokens = labels[i][resp_mask]

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
                            'train/expected_length_loss': avg_length_loss.item(),
                            'train/surrounding_ctx_loss': avg_surrounding_ctx_loss.item(),
                            'train/valid_ctx': valid_ctx,
                            'train/base_loss': base_loss.item(),
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
        sample_mask = labels[sample_idx] != -100
        pred_tokens = predicted_ids[sample_idx][sample_mask]
        predicted_text = self.processing_class.tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
        ground_truth_text = self.processing_class.tokenizer.decode(
            labels[sample_idx][sample_mask], skip_special_tokens=True
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
