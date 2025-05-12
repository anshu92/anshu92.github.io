---
title: "A Gentle Introduction to Transformer Models"
date: 2025-05-12T12:00:00-04:00 # Use current date/time & timezone
draft: false # Set to 'true' while writing, 'false' to publish
tags: ["NLP", "Deep Learning", "Transformers", "Attention"]
categories: ["Concepts"]
series: ["NLP Basics"] # Optional: group related posts
summary: "Explaining the core concepts behind the Transformer architecture, including self-attention."
thumbnail: "/images/transformer-thumb.png" # Optional: Add to static/images/
showAuthor: true # Override default from params.toml if needed
showTableOfContents: true # Show ToC for this post
math: true # Enable math rendering JUST for this page
mermaid: true # Enable Mermaid diagrams JUST for this page
---

## Introduction

Transformers have revolutionized Natural Language Processing...

## The Attention Mechanism

The key innovation is the self-attention mechanism. The formula is:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Where $Q$ (Query), $K$ (Key), and $V$ (Value) are learned linear projections of the input embeddings.

## Code Example (Python)

```python
import torch
import torch.nn as nn

# Simplified Self-Attention Example
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # ... (Implementation details) ...

    def forward(self, value, key, query, mask):
        # ... (Attention calculation) ...
        return out