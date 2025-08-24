# src/model.py
import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional
import config

class SLiQ(layers.Layer):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.where(x >= 0, x, x + 0.5 * tf.pow(x, 2))

class Head(layers.Layer):
    def __init__(self, head_size: int, **kwargs):
        super().__init__(**kwargs)
        self.key = layers.Dense(head_size, use_bias=False)
        self.query = layers.Dense(head_size, use_bias=False)
        self.value = layers.Dense(head_size, use_bias=False)
        self.dropout = layers.Dropout(config.DROPOUT)
        # tril should be a constant, not a variable/weight
        self.tril = tf.linalg.band_part(tf.ones((config.BLOCK_SIZE, config.BLOCK_SIZE)), -1, 0)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        k = self.key(x)
        q = self.query(x)
        
        wei = tf.matmul(q, k, transpose_b=True) * (tf.cast(C, tf.float32)**-0.5)
        
        # Slice the constant tril tensor
        tril_slice = self.tril[:T, :T]
        
        wei = tf.where(tril_slice == 0, float('-inf'), wei)
        wei = tf.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = tf.matmul(wei, v)
        return out

class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads: int, head_size: int, **kwargs):
        super().__init__(**kwargs)
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = layers.Dense(config.N_EMBED)
        self.dropout = layers.Dropout(config.DROPOUT)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(layers.Layer):
    def __init__(self, n_embed: int, **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential([
            layers.Dense(4 * n_embed),
            SLiQ(),
            layers.Dense(n_embed),
            layers.Dropout(config.DROPOUT),
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.net(x)

class Block(layers.Layer):
    def __init__(self, n_embed: int, n_head: int, **kwargs):
        super().__init__(**kwargs)
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(tf.keras.Model):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding_table = layers.Embedding(vocab_size, config.N_EMBED)
        self.position_embedding_table = layers.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.blocks = tf.keras.Sequential([Block(config.N_EMBED, n_head=config.N_HEAD) for _ in range(config.N_LAYER)])
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.lm_head = layers.Dense(vocab_size)

    def call(self, idx: tf.Tensor, targets: Optional[tf.Tensor] = None):
        B, T = tf.shape(idx)[0], tf.shape(idx)[1]
        
        tok_emb = self.token_embedding_table(idx)
        
        # Create positional embeddings
        pos = tf.range(start=0, limit=T, delta=1)
        pos_emb = self.position_embedding_table(pos)
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Reshape for sparse categorical crossentropy
            logits_flat = tf.reshape(logits, [-1, logits.shape[-1]])
            targets_flat = tf.reshape(targets, [-1])
            
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets_flat, logits_flat, from_logits=True)
            loss = tf.reduce_mean(loss)
            
        return logits, loss