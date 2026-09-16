import torch
import torch.nn as nn
import numpy as np

class SmolVLMTokenizer:
    """
    Subword tokenizer and vocabulary for Dobot VLA task instructions.
    Tokenizes natural language strings like:
    'pick up the red cube and place it on the green platform'
    into discrete token IDs with padding and attention masks.
    """
    def __init__(self):
        # Dedicated manipulation vocabulary with special tokens
        self.special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>', '<image>', '</image>']
        self.words = [
            'pick', 'up', 'the', 'and', 'place', 'it', 'on', 'platform',
            'push', 'towards', 'cube', 'box', 'to', 'target',
            'red', 'blue', 'yellow', 'green', 'purple', 'orange', 'cyan',
            'reach', 'grasp', 'lift', 'drop', 'retract'
        ]
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens + self.words)}
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        self.pad_id = self.vocab['<pad>']
        self.bos_id = self.vocab['<bos>']
        self.eos_id = self.vocab['<eos>']

    def encode(self, text, max_len=16):
        clean = text.lower().replace('.', '').replace(',', '').strip().split()
        ids = [self.bos_id]
        for w in clean:
            ids.append(self.vocab.get(w, self.vocab['<unk>']))
        ids.append(self.eos_id)
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)

class SmolVLAPairedEmbedding(nn.Module):
    """
    SmolVLM / SmolVLA Paired Token-Patch Fusion Layer:
    - Text: Tokens embedded to d_model via Language Embedding Layer.
    - Image: 64x64 top camera divided into 16x16 patches = 256 visual tokens with 2D Positional Embeddings.
    - Token-Patch Cross-Correlation: Computes dense pairwise dot-product alignment between
      each language token (e.g. 'red', 'platform') and each 4x4 visual patch.
    """
    def __init__(self, vocab_size=32, d_model=128, text_len=16, num_patches=256):
        super().__init__()
        self.d_model = d_model
        self.text_len = text_len
        self.num_patches = num_patches

        # 1. Language Token Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.text_pos_embed = nn.Parameter(torch.randn(1, text_len, d_model) * 0.02)

        # 2. Vision Patch Embedding (4x4 patches on 64x64 overhead camera)
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=4, stride=4)
        self.vis_pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        # 3. Projection layers for pairing / cross-modal affinity
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / np.sqrt(d_model)

    def forward(self, text_ids, img):
        B = img.size(0)

        # Text tokens: [B, text_len, d_model]
        text_feats = self.token_embed(text_ids) + self.text_pos_embed

        # Visual patch tokens: [B, 256, d_model]
        patches = self.patch_embed(img).flatten(2).permute(0, 2, 1)
        vis_feats = patches + self.vis_pos_embed

        # Paired Token-to-Patch Attention Matrix: [B, text_len, num_patches]
        # Measures how strongly each word in the prompt attends to each visual patch
        q = self.q_proj(text_feats) # [B, text_len, d_model]
        k = self.k_proj(vis_feats)   # [B, num_patches, d_model]
        
        affinity = torch.bmm(q, k.transpose(1, 2)) * self.scale # [B, text_len, 256]
        paired_weights = torch.softmax(affinity, dim=-1)         # Softmax over 256 patches

        return text_feats, vis_feats, paired_weights

if __name__ == '__main__':
    tok = SmolVLMTokenizer()
    prompt = 'pick up the red cube and place it on the green platform'
    ids = tok.encode(prompt).unsqueeze(0)
    img = torch.randn(1, 3, 64, 64)

    embedder = SmolVLAPairedEmbedding(vocab_size=len(tok.vocab), d_model=128)
    t_feats, v_feats, paired_map = embedder(ids, img)
    print(f'Text Feats: {t_feats.shape}')
    print(f'Vision Feats: {v_feats.shape}')
    print(f'Paired Token-Patch Heatmap Shape: {paired_map.shape} ([Batch, Tokens=16, Patches=256])')
    print('SmolVLA Paired Token-Patch Architecture Initialized Successfully!')
