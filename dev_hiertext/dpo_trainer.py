#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online DPO (Direct Preference Optimization) Trainer
SFT学習完了後のファインチューニング用
"""

import os
import random

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from peft import PeftModel

from custom_trainer import CustomTrainer


class DPOTrainer(CustomTrainer):
    """Online DPO選好チューニングを追加するTrainer（SFT後に使用）"""

    def __init__(
        self,
        *args,
        dpo_weight=0.1,
        dpo_beta=0.1,
        dpo_num_candidates=3,
        dpo_num_samples=2,
        dpo_freq=100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dpo_weight = dpo_weight
        self.dpo_beta = dpo_beta
        self.dpo_num_candidates = dpo_num_candidates
        self.dpo_num_samples = dpo_num_samples
        self.dpo_freq = dpo_freq

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # surrounding_texts / masked_image_paths は super() 内で pop されるため先に取得
        surrounding_texts = inputs.get('surrounding_texts')
        masked_image_paths = inputs.get('masked_image_paths')
        labels = inputs.get('labels')
        current_step = self.state.global_step if self.state else 0

        result = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        loss, outputs = result

        avg_dpo_loss = loss.new_zeros(())
        if current_step % self.dpo_freq == 0 and self.dpo_weight > 0 and surrounding_texts:
            avg_dpo_loss = self._compute_dpo_loss(
                model, inputs, labels, surrounding_texts, masked_image_paths
            )
            loss = loss + self.dpo_weight * avg_dpo_loss

        if current_step % self.compute_metrics_freq == 0:
            self.log({'train/dpo_loss': avg_dpo_loss.item()})

        return (loss, outputs) if return_outputs else loss

    def _compute_sequence_logprob(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, response_start):
        """response部分の token log prob の合計を計算"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        log_probs = F.log_softmax(outputs.logits[0], dim=-1)
        response_log_prob = log_probs.new_zeros(())
        for t in range(response_start, input_ids.shape[1] - 1):
            token_id = input_ids[0, t + 1]
            response_log_prob = response_log_prob + log_probs[t, token_id]
        return response_log_prob

    def _compute_dpo_loss(self, model, inputs, labels, surrounding_texts, masked_image_paths=None):
        """Online DPO loss: 候補生成 → Qwen3-VL-Embeddingスコアリング → DPO loss計算"""
        raw_model = model.module if hasattr(model, 'module') else model
        processor = self.processing_class
        device = labels.device
        batch_size = labels.shape[0]
        eos_id = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')

        embed_model, embed_proc = self._get_embed_model(device)

        dpo_loss_accum = torch.zeros((), device=device)
        valid_dpo = 0

        candidate_indices = random.sample(range(batch_size), min(self.dpo_num_samples, batch_size))

        for i in candidate_indices:
            if not surrounding_texts or not surrounding_texts[i]:
                continue

            # per-sampleのpixel_values範囲を計算（各サンプルに2画像）
            n_patches_before = sum(int(t) * int(h) * int(w) for t, h, w in inputs['image_grid_thw'][: 2 * i])
            n_patches_this = sum(int(t) * int(h) * int(w) for t, h, w in inputs['image_grid_thw'][2 * i : 2 * i + 2])
            sample_pv = inputs['pixel_values'][n_patches_before : n_patches_before + n_patches_this]
            sample_thw = inputs['image_grid_thw'][2 * i : 2 * i + 2]

            # promptの終端位置を特定
            resp_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(resp_positions) == 0:
                continue
            response_start = resp_positions[0].item()

            # N候補を生成（eval modeで推論、その後train modeに戻す）
            gen_inputs = {
                'input_ids': inputs['input_ids'][i : i + 1, :response_start],
                'attention_mask': inputs['attention_mask'][i : i + 1, :response_start],
                'pixel_values': sample_pv,
                'image_grid_thw': sample_thw,
            }
            raw_model.eval()
            try:
                with torch.inference_mode():
                    generated = raw_model.generate(
                        **gen_inputs,
                        max_new_tokens=32,
                        do_sample=True,
                        temperature=0.9,
                        num_return_sequences=self.dpo_num_candidates,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=eos_id,
                    )
            finally:
                raw_model.train()

            prompt_len = gen_inputs['input_ids'].shape[1]
            candidates_text = [
                processor.tokenizer.decode(generated[k, prompt_len:], skip_special_tokens=True).strip()
                for k in range(len(generated))
            ]
            unique_texts = list(dict.fromkeys([t for t in candidates_text if t]))
            if len(unique_texts) < 2:
                continue

            # Qwen3-VL-Embedding で文脈スコアリング → chosen/rejected 選択
            with torch.no_grad():
                # 候補テキストをエンコード
                cand_enc = embed_proc(
                    text=unique_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=64,
                ).to(device)
                cand_embeds = F.normalize(
                    embed_model(**cand_enc).last_hidden_state[:, -1, :].float(),
                    dim=-1,
                )  # [N, 4096]

                # 周辺テキストをエンコード → mean
                ctx_texts = surrounding_texts[i][:8]
                ctx_enc = embed_proc(
                    text=ctx_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=64,
                ).to(device)
                ctx_text_embed = F.normalize(
                    embed_model(**ctx_enc).last_hidden_state[:, -1, :].float().mean(dim=0),
                    dim=-1,
                )  # [4096]

                # 周辺画像をエンコード（利用可能な場合）
                ctx_embed = ctx_text_embed
                if (masked_image_paths and masked_image_paths[i]
                        and os.path.exists(masked_image_paths[i])):
                    img = PILImage.open(masked_image_paths[i]).convert('RGB')
                    img_enc = embed_proc(images=[img], return_tensors='pt').to(device)
                    img_embed = F.normalize(
                        embed_model(**img_enc).last_hidden_state.mean(dim=1).squeeze(0).float(),
                        dim=-1,
                    )  # [4096]
                    ctx_embed = F.normalize(ctx_text_embed + img_embed, dim=-1)  # [4096]

                scores = (cand_embeds * ctx_embed).sum(dim=-1)  # [N]

            best_idx = scores.argmax().item()
            worst_idx = scores.argmin().item()
            if best_idx == worst_idx:
                continue

            chosen_text = unique_texts[best_idx]
            rejected_text = unique_texts[worst_idx]

            # chosen/rejectedシーケンスを構築（prompt + response + eos）
            def build_seq(response_text):
                resp_ids = processor.tokenizer.encode(response_text, add_special_tokens=False)
                resp_tensor = torch.tensor(resp_ids + [eos_id], dtype=torch.long, device=device).unsqueeze(0)
                seq_ids = torch.cat([inputs['input_ids'][i : i + 1, :response_start], resp_tensor], dim=1)
                seq_mask = torch.ones(seq_ids.shape, dtype=torch.long, device=device)
                return seq_ids, seq_mask

            chosen_ids, chosen_mask = build_seq(chosen_text)
            rejected_ids, rejected_mask = build_seq(rejected_text)

            # current model (adapter ON) の log prob（勾配あり）
            log_p_chosen = self._compute_sequence_logprob(
                raw_model, chosen_ids, chosen_mask, sample_pv, sample_thw, response_start
            )
            log_p_rejected = self._compute_sequence_logprob(
                raw_model, rejected_ids, rejected_mask, sample_pv, sample_thw, response_start
            )

            # reference model (adapter OFF) の log prob（勾配なし）
            if isinstance(raw_model, PeftModel):
                with raw_model.disable_adapter():
                    with torch.no_grad():
                        log_ref_chosen = self._compute_sequence_logprob(
                            raw_model, chosen_ids, chosen_mask, sample_pv, sample_thw, response_start
                        )
                        log_ref_rejected = self._compute_sequence_logprob(
                            raw_model, rejected_ids, rejected_mask, sample_pv, sample_thw, response_start
                        )
            else:
                raw_model.disable_adapters()
                try:
                    with torch.no_grad():
                        log_ref_chosen = self._compute_sequence_logprob(
                            raw_model, chosen_ids, chosen_mask, sample_pv, sample_thw, response_start
                        )
                        log_ref_rejected = self._compute_sequence_logprob(
                            raw_model, rejected_ids, rejected_mask, sample_pv, sample_thw, response_start
                        )
                finally:
                    raw_model.enable_adapters()

            # DPO loss: -log σ(β * ((log πθ(chosen) - log πref(chosen)) - (log πθ(rejected) - log πref(rejected))))
            chosen_reward = self.dpo_beta * (log_p_chosen - log_ref_chosen)
            rejected_reward = self.dpo_beta * (log_p_rejected - log_ref_rejected)
            dpo_loss_accum = dpo_loss_accum + (-F.logsigmoid(chosen_reward - rejected_reward))
            valid_dpo += 1

        if valid_dpo > 0:
            return dpo_loss_accum / valid_dpo
        return torch.zeros((), device=device)
