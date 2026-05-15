import torch
from torch import nn, Tensor
from math import sqrt

from .blocks import LayerNorm, Transformer


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype

    def forward(self, text: Tensor):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

class DeepPromptCLIPTextEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            num_tokens: int = 8,  # DE-CLIP: Number of deep prompt tokens
            prompt_depth: int = 12,  # DE-CLIP: How many layers deep to inject
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.num_tokens = num_tokens
        self.prompt_depth = prompt_depth

        # --- DE-CLIP: Learnable Deep Prompts ---
        # Shape: [Depth, Batch (1), Num_Tokens, Embedding_Dim]
        self.deep_prompts = nn.Parameter(
            torch.empty(prompt_depth, 1, num_tokens, transformer_width)
        )
        nn.init.normal_(self.deep_prompts, std=1 / sqrt(transformer_width))

        # Standard token embedding
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        # --- ATTENTION MASKS ---
        # Standard mask for normal text encoding
        self.standard_mask = self.build_attention_mask(context_length)
        # Extended mask for when deep prompts are prepended
        self.extended_mask = self.build_extended_attention_mask(context_length, num_tokens)

        # Initialize Transformer (We pass None to mask initially, handled in forward pass)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=None,
        )

        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def build_attention_mask(self, context_length):
        """Standard causal mask for original text"""
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def build_extended_attention_mask(self, context_length, num_tokens):
        """
        Extended mask combining deep prompts and text:
        - Prompts can see other prompts.
        - Text can see all prompts.
        - Text can only see previous text (causal).
        - Prompts CANNOT see text.
        """
        total_len = context_length + num_tokens
        mask = torch.empty(total_len, total_len)
        mask.fill_(float("-inf"))

        # Prompts see prompts
        mask[:num_tokens, :num_tokens] = 0.0
        # Text sees prompts
        mask[num_tokens:, :num_tokens] = 0.0

        # Text sees text causally
        text_mask = torch.empty(context_length, context_length)
        text_mask.fill_(float("-inf"))
        text_mask.triu_(1)
        mask[num_tokens:, num_tokens:] = text_mask

        return mask

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype

    def forward(self, text: Tensor):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # --- DE-CLIP DEEP PROMPT INJECTION LOOP ---
        for i, resblock in enumerate(self.transformer.resblocks):
            if i < self.prompt_depth:
                # 1. Expand current layer's deep prompts for the batch size
                dp = self.deep_prompts[i].expand(x.shape[1], -1, -1)  # [Batch, num_tokens, Dim]
                dp = dp.permute(1, 0, 2).to(x.dtype)  # [num_tokens, Batch, Dim]

                # 2. Prepend deep prompts to the text sequence
                x = torch.cat([dp, x], dim=0)

                # 3. Assign the extended mask & run transformer block
                resblock.attn_mask = self.extended_mask.to(device=x.device, dtype=x.dtype)
                x = resblock(x)

                # 4. Remove the processed prompts before the next layer
                # (Prevents sequence from growing infinitely & aligns the EoT token)
                x = x[self.num_tokens:, :, :]
            else:
                # Standard forward pass for deeper layers without prompts
                resblock.attn_mask = self.standard_mask.to(device=x.device, dtype=x.dtype)
                x = resblock(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Extract features using the End-Of-Text (EoT) token index
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x