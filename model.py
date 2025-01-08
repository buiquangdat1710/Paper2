import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModel
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_ckpt_c = 'neulab/codebert-c'
model_ckpt_cpp = 'neulab/codebert-cpp'
model_ckpt_t5 = 'Salesforce/codet5p-110m-embedding'
model_ckpt_unixcoder = 'microsoft/unixcoder-base'
model_codesage_small = 'codesage/codesage-small'
model_roberta = 'FacebookAI/roberta-base'
model_name = model_ckpt_t5
tokenizer = AutoTokenizer.from_pretrained(model_name)

file_path = 'bigvul.csv'
data = pd.read_csv(file_path)
print(len(data))
data = data[['code', 'label']]
data['label'].value_counts()
print(data['label'].value_counts())

comment_regex = r'(//[^\n]*|\/\*[\s\S]*?\*\/)'
newline_regex = '\n{1,}'
whitespace_regex = '\s{2,}'

def data_cleaning(inp, pat, rep):
    return re.sub(pat, rep, inp)

data['truncated_code'] = (data['code'].apply(data_cleaning, args=(comment_regex, ''))
                                      .apply(data_cleaning, args=(newline_regex, ' '))
                                      .apply(data_cleaning, args=(whitespace_regex, ' '))
                         )
# remove all data points that have more than 15000 characters
data = data[data['truncated_code'].str.len() < 12000]
print(f"Number of samples after length filtering: {len(data)}")

from sklearn.model_selection import train_test_split
X_train, X_test_valid, y_train, y_test_valid = train_test_split(data.loc[:, data.columns != 'label'],
                                                                data['label'],
                                                                train_size=0.8,
                                                                stratify=data['label']
                                                               )
X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid.loc[:, X_test_valid.columns != 'label'],
                                                    y_test_valid,
                                                    test_size=0.5,
                                                    stratify=y_test_valid)

data_train = X_train
data_train['label'] = y_train
data_test = X_test
data_test['label'] = y_test
data_valid = X_valid
data_valid['label'] = y_valid

from datasets import Dataset, DatasetDict
dts = DatasetDict()
dts['train'] = Dataset.from_pandas(data_train)
dts['test'] = Dataset.from_pandas(pd.concat([data_test, data_valid]))
dts['valid'] = Dataset.from_pandas(pd.concat([data_test, data_valid]))
print(dts)

def tokenizer_func(examples):
    result = tokenizer(examples['truncated_code'], padding=True, truncation=True,max_length=512 )
    return result

dts = dts.map(tokenizer_func,
             batched=True,
             batch_size=4
             )

dts.set_format('torch')
dts.rename_column('label', 'labels')
dts = dts.remove_columns(['code', 'truncated_code', '__index_level_0__'])

import torch.nn as nn
import torch
from transformers import AutoModel

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, dropout=0.05):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)  
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(1, 1, 1, dim))  
        self.chunk_size = 128  
        
        self.cos_cached = None
        self.sin_cached = None
        self.seq_len_cached = None
        
        self.register_forward_hook(self._log_rotation_stats)
        
    def _get_rotation(self, seq_len):
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}")

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]

        return self.cos_cached, self.sin_cached

    def _log_rotation_stats(self, module, input, output):
        cos, sin = output
        with torch.no_grad():
            cos_norm = torch.norm(cos)
            sin_norm = torch.norm(sin)
            if torch.isnan(cos_norm) or torch.isnan(sin_norm):
                logging.warning(f"NaN detected in RoPE: cos_norm={cos_norm}, sin_norm={sin_norm}")
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
            
        chunks = x.chunk(math.ceil(x.shape[1] / self.chunk_size), dim=1)
        rotated_chunks = []
        
        for chunk in chunks:
            cos, sin = self._get_rotation(chunk.shape[1])
            
            cos = self.dropout(cos) * self.gate
            sin = self.dropout(sin) * self.gate
            
            rotated_chunks.append((cos[:, :chunk.shape[1]], sin[:, :chunk.shape[1]]))
            
        cos = torch.cat([c[0] for c in rotated_chunks], dim=1)
        sin = torch.cat([c[1] for c in rotated_chunks], dim=1)
        
        return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    original_x = x
    x_reshape = x.unsqueeze(-2)
    x_1, x_2 = x_reshape.chunk(2, dim=-1)
    
    x_rotated = torch.cat(
        (
            x_1 * cos - x_2 * sin,
            x_2 * cos + x_1 * sin,
        ),
        dim=-1,
    ).squeeze(-2)
    
    return x_rotated + 0.1 * original_x

class CodeBertModel(nn.Module):
    def __init__(self,
                 max_seq_length: int = 512,
                 chunk_size: int = 512,
                 padding_idx: int = 0,
                 model_ckpt: str = '',
                 num_heads: int = 4,
                 dropout: float = 0.15,
                 **from_pretrained_kwargs):
        super().__init__()
        self.embedding_model = AutoModel.from_pretrained(model_ckpt, trust_remote_code=True).to(device)

        dict_config = self.embedding_model.config.to_dict()
        embed_dim = dict_config.get('hidden_size', 768)
        
        self.scale = 1. / math.sqrt(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.rope = RotaryEmbedding(dim=embed_dim//2, max_seq_len=max_seq_length, dropout=dropout)
        
        # Improved transformer with better configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2*embed_dim,  # Increased FFN size
            dropout=dropout,
            batch_first=True,
            activation=F.gelu  # Using GELU activation
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=5,
            norm=nn.LayerNorm(embed_dim)  # Added layer normalization
        )

        # Enhanced FFN with skip connection
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2)
        )

        # Weighted cross entropy with label smoothing
        self.loss_func = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 6.0]).to(device),
            label_smoothing=0.1
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        
        # Apply layer norm and dropout
        hidden_states = F.layer_norm(hidden_states, hidden_states.shape[2:])
        hidden_states = self.dropout(hidden_states * self.scale)
        
        batch_size, seq_len, embed_dim = hidden_states.shape
        cos, sin = self.rope(hidden_states, seq_len=seq_len)
        
        hidden_states = apply_rotary_pos_emb(hidden_states, cos, sin) 
        # Add residual connection
        residual = hidden_states 
        # Apply transformer with improved masking
        mask = attention_mask == 0
        encoded = self.transformer_encoder(hidden_states, src_key_padding_mask=mask)
        encoded = encoded + residual  # Residual connection  
        # Weighted pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        weights = F.softmax(self.ffn[1](encoded), dim=1)  # Learn attention weights
        weights = weights * mask_expanded
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled_output = torch.sum(encoded * weights, dim=1)
        logits = self.ffn(pooled_output)
        if labels is not None:
            loss = self.loss_func(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
def compute_metrics(eval_pred):
    y_pred, y_true = np.argmax(eval_pred.predictions, -1), eval_pred.label_ids
    return {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)}
model = CodeBertModel(model_ckpt = model_name, max_seq_length=512 , chunk_size = 512, num_heads=8).to(device)
from transformers import DataCollatorWithPadding
import os
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
directory = "modelsave"
if not os.path.exists(directory):
    os.makedirs(directory)
training_arguments = TrainingArguments(
    output_dir = './modelsave',
    evaluation_strategy = 'steps',
    eval_steps = 100,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    gradient_accumulation_steps = 16,
    learning_rate = 2e-4,
    num_train_epochs = 15,
    warmup_ratio = 0.1,
    lr_scheduler_type = 'cosine',
    weight_decay = 0.01,
    logging_strategy = 'steps',
    logging_steps = 50,
    save_strategy = 'steps',
    save_steps = 500,
    save_total_limit = 2,
    load_best_model_at_end = True,
    metric_for_best_model = 'f1',
    fp16 = True,
    optim = 'adamw_torch',
    report_to = 'none'
)
trainer = Trainer(model=model,
                  data_collator=data_collator,
                  args=training_arguments,
                  train_dataset=dts['train'],
                  eval_dataset=dts['valid'],
                  compute_metrics=compute_metrics,
                 )
trainer.train()
check = trainer.predict(dts['test'])
print(compute_metrics(check))
