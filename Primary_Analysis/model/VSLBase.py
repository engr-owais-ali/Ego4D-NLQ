import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from model.layers import (
    Embedding,
    VisualProjection,
    FeatureEncoder,
    CQAttention,
    CQConcatenate,
    ConditionedPredictor,
    BertEmbedding,
)


def build_optimizer_and_scheduler(model, configs):
    no_decay = ["bias", "layer_norm", "LayerNorm"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        configs.num_train_steps * configs.warmup_proportion,
        configs.num_train_steps,
    )
    return optimizer, scheduler


class VSLBase(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLBase, self).__init__()
        self.configs = configs

        # project raw video features into model-dim
        self.video_affine = VisualProjection(
            visual_dim=configs.video_feature_dim,
            dim=configs.dim,
            drop_rate=configs.drop_rate,
        )

        # 4-layer convolutional feature encoder (video & query)
        self.feature_encoder = FeatureEncoder(
            dim=configs.dim,
            num_heads=configs.num_heads,
            kernel_size=7,
            num_layers=4,
            max_pos_len=configs.max_pos_len,
            drop_rate=configs.drop_rate,
        )

        # cross-modal fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat    = CQConcatenate(dim=configs.dim)

        # final span predictor
        self.predictor = ConditionedPredictor(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len,
            predictor=configs.predictor,
        )

        # text embedding (either BERT or word+char)
        if configs.predictor == "bert":
            # project BERT’s 768-dim back to configs.dim
            self.query_affine = nn.Linear(768, configs.dim)
            self.init_parameters()
            self.embedding_net = BertEmbedding(configs.text_agnostic)
        else:
            self.embedding_net = Embedding(
                num_words=configs.word_size,
                num_chars=configs.char_size,
                out_dim=configs.dim,
                word_dim=configs.word_dim,
                char_dim=configs.char_dim,
                word_vectors=word_vectors,
                drop_rate=configs.drop_rate,
            )
            self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        # project & encode video
        vf = self.video_affine(video_features)
        vf = self.feature_encoder(vf, mask=v_mask)

        # embed & encode query
        if self.configs.predictor == "bert":
            qf = self.embedding_net(word_ids)           # (B, L_q, 768)
            qf = self.query_affine(qf)                  # → (B, L_q, dim)
        else:
            qf = self.embedding_net(word_ids, char_ids) # (B, L_q, dim)
        qf = self.feature_encoder(qf, mask=q_mask)

        # fuse
        f = self.cq_attention(vf, qf, v_mask, q_mask)
        f = self.cq_concat(f, qf, q_mask)

        # span prediction
        start_logits, end_logits = self.predictor(f, mask=v_mask)
        return start_logits, end_logits

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(
            start_logits=start_logits,
            end_logits=end_logits,
        )

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
        )