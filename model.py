import tensorflow as tf
from tensorflow.keras import layers

# ════════════════════════════════════════════════════════════════════════════
# 1. PatchEmbedding with Adaptive Patch Weighting (Equation 1 & 2 in paper)
# ════════════════════════════════════════════════════════════════════════════
class PatchEmbedding(layers.Layer):
    """
    Conv2D projection + learnable patch_weight + LayerNorm.
    Initialize patch_weight to ones (paper Eq. 1).
    """
    def __init__(self, patch_size, stride, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride     = stride
        self.embed_dim  = embed_dim

        self.proj = layers.Conv2D(
            embed_dim, kernel_size=patch_size,
            strides=stride, padding="same"
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        # patch_weight: (1, num_patches, embed_dim) – broadcast over batch
        self.patch_weight = None   # Assigned in build()

    def build(self, input_shape):
        # Compute the spatial token count dynamically.
        # Learn one channel-weight vector shared across tokens.
        self.patch_weight = self.add_weight(
            name="patch_weight",
            shape=(1, 1, self.embed_dim),   # Broadcast to (B, N, C).
            initializer="ones",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # (B, H, W, C) → (B, H', W', embed_dim)
        x = self.proj(x)
        H, W = tf.shape(x)[1], tf.shape(x)[2]

        # Flatten → (B, N, embed_dim)
        x_flat = tf.reshape(x, (tf.shape(x)[0], H * W, self.embed_dim))

        # Adaptive patch weighting – Eq.1: X_flat_weighted = X_flat ⊙ W_patch
        x_flat = x_flat * self.patch_weight

        # LayerNorm – Eq.2
        x_norm = self.norm(x_flat)
        return x_norm, H, W


# ════════════════════════════════════════════════════════════════════════════
# 2. Relative Positional Encoding 2D  (Eq.6 in paper)
# ════════════════════════════════════════════════════════════════════════════
class RelativePositionBias2D(layers.Layer):
    """
    Learn a relative positional bias for each patch pair (i, j).
    B_RPE[i,j,k] = B_height[yi-yj + Sy, k] + B_width[xi-xj + Sx, k]
    """
    def __init__(self, num_heads, max_relative_position=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads           = num_heads
        self.max_relative_pos    = max_relative_position
        # 2*max_rel_pos - 1 distinct relative distances per axis
        table_size = 2 * max_relative_position - 1
        self.relative_height = self.add_weight(
            name="rel_height", shape=(table_size, num_heads),
            initializer="zeros", trainable=True
        )
        self.relative_width  = self.add_weight(
            name="rel_width",  shape=(table_size, num_heads),
            initializer="zeros", trainable=True
        )

    def call(self, H, W):
        """Returns RPE bias of shape (num_heads, H*W, H*W)."""
        coords_h = tf.range(H)
        coords_w = tf.range(W)
        # meshgrid → (H, W) → flatten → (H*W,)
        grid_h, grid_w = tf.meshgrid(coords_h, coords_w, indexing='ij')
        grid_h = tf.reshape(grid_h, (-1,))  # (N,)
        grid_w = tf.reshape(grid_w, (-1,))

        # relative coords → clipped to [-max+1, max-1]
        S = self.max_relative_pos - 1
        rel_h = tf.clip_by_value(
            grid_h[:, None] - grid_h[None, :], -(S), S
        ) + S   # (N, N), range [0, 2*max-2]
        rel_w = tf.clip_by_value(
            grid_w[:, None] - grid_w[None, :], -(S), S
        ) + S

        # Lookup biases – shapes (N, N, num_heads)
        bias_h = tf.gather(self.relative_height, rel_h)
        bias_w = tf.gather(self.relative_width,  rel_w)
        bias   = bias_h + bias_w   # (N, N, num_heads)

        # → (num_heads, N, N)
        bias = tf.transpose(bias, (2, 0, 1))
        return bias


# ════════════════════════════════════════════════════════════════════════════
# 3. Hausdorff-bias Multi-Head Attention  (Eq.3, 4, 7 in paper)
# ════════════════════════════════════════════════════════════════════════════
class HausdorffMultiHeadAttention(layers.Layer):
    """
    Standard MHA + Hausdorff spatial bias + RPE bias.

    Attention score pipeline (paper Eq.4 & 7):
        S        = Q @ K^T / sqrt(d_k)
        S'       = S + B_Hausdorff          (Eq.4)
        S''      = S' + B_RPE               (Eq.7)
        output   = softmax(S'') @ V
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = layers.Dense(embed_dim)
        self.k_proj = layers.Dense(embed_dim)
        self.v_proj = layers.Dense(embed_dim)
        self.out    = layers.Dense(embed_dim)

    @staticmethod
    def _hausdorff_bias(H, W):
        """
        Pre-compute Hausdorff bias from Euclidean distances (Eq.3).
        B_ij = -d(pi, pj) / (max(D) + eps)
        Returns (1, 1, N, N) tensor (broadcast over batch & heads).
        """
        coords_h = tf.cast(tf.range(H), tf.float32)
        coords_w = tf.cast(tf.range(W), tf.float32)
        gh, gw   = tf.meshgrid(coords_h, coords_w, indexing='ij')
        gh = tf.reshape(gh, (-1,))  # (N,)
        gw = tf.reshape(gw, (-1,))

        # Pairwise Euclidean distances → (N, N)
        dh = gh[:, None] - gh[None, :]
        dw = gw[:, None] - gw[None, :]
        D  = tf.sqrt(dh**2 + dw**2)

        eps   = 1e-6
        B_hd  = -D / (tf.reduce_max(D) + eps)   # Eq.3
        # (1, 1, N, N) for broadcasting
        return B_hd[None, None, :, :]

    def call(self, x, rpe_bias=None, H=None, W=None):
        B  = tf.shape(x)[0]
        N  = tf.shape(x)[1]
        nh = self.num_heads
        hd = self.head_dim

        Q = tf.reshape(self.q_proj(x), (B, N, nh, hd))
        K = tf.reshape(self.k_proj(x), (B, N, nh, hd))
        V = tf.reshape(self.v_proj(x), (B, N, nh, hd))

        # (B, nh, N, hd)
        Q = tf.transpose(Q, (0,2,1,3))
        K = tf.transpose(K, (0,2,1,3))
        V = tf.transpose(V, (0,2,1,3))

        # Raw attention scores
        S = tf.matmul(Q, K, transpose_b=True) * self.scale   # (B, nh, N, N)

        # Hausdorff bias – Eq.4
        if H is not None and W is not None:
            S = S + tf.cast(self._hausdorff_bias(H, W), S.dtype)

        # RPE bias – Eq.7
        if rpe_bias is not None:
            # rpe_bias: (nh, N, N) → (1, nh, N, N)
            S = S + tf.cast(rpe_bias[None], S.dtype)

        # Stabilize & softmax
        S = S - tf.reduce_max(S, axis=-1, keepdims=True)
        attn = tf.nn.softmax(S)

        out = tf.matmul(attn, V)               # (B, nh, N, hd)
        out = tf.transpose(out, (0,2,1,3))     # (B, N, nh, hd)
        out = tf.reshape(out, (B, N, nh*hd))
        return self.out(out)


# ════════════════════════════════════════════════════════════════════════════
# 4. MixFFN with DepthwiseConv2D  (Eq.5 in paper)
# ════════════════════════════════════════════════════════════════════════════
class MixFFN(layers.Layer):
    """
    fc1 → GELU → reshape → DepthwiseConv2D → flatten → LN → fc2
    DepthwiseConv2D adds local spatial context per channel (Eq.5).
    """
    def __init__(self, embed_dim, expand_ratio=4, **kwargs):
        super().__init__(**kwargs)
        exp = embed_dim * expand_ratio
        self.fc1   = layers.Dense(exp)
        self.act   = layers.Activation("gelu")
        self.dwconv = layers.DepthwiseConv2D(
            kernel_size=3, padding="same", use_bias=True
        )
        self.norm  = layers.LayerNormalization(epsilon=1e-6)
        self.fc2   = layers.Dense(embed_dim)

    def call(self, x, H, W):
        u = self.act(self.fc1(x))                           # (B, N, exp)
        # Reshape to spatial for depthwise conv
        B   = tf.shape(u)[0]
        exp = tf.shape(u)[-1]
        u_s = tf.reshape(u, (B, H, W, exp))                 # (B,H,W,exp)
        u_s = self.dwconv(u_s)                              # Eq.5
        u   = tf.reshape(u_s, (B, H*W, exp))
        u   = self.norm(u)
        return self.fc2(u)


# ════════════════════════════════════════════════════════════════════════════
# 5. TransformerEncoderLayer – ties everything together
# ════════════════════════════════════════════════════════════════════════════
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, max_rel_pos=32, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.attn  = HausdorffMultiHeadAttention(embed_dim, num_heads)
        self.ffn   = MixFFN(embed_dim)
        self.rpe   = RelativePositionBias2D(num_heads, max_rel_pos)
        self.drop  = layers.Dropout(0.1)

    def call(self, x, H, W, training=False):
        rpe_bias = self.rpe(H, W)
        # Attention sub-layer
        x = x + self.drop(
            self.attn(self.norm1(x), rpe_bias=rpe_bias, H=H, W=W),
            training=training
        )
        # FFN sub-layer
        x = x + self.drop(
            self.ffn(self.norm2(x), H, W),
            training=training
        )
        return x

class MLSRSSegFormer(tf.keras.Model):
    STAGE_CFG = [
        (7, 4,  32,  1, 2),
        (3, 2,  64,  2, 2),
        (3, 2, 160,  5, 2),
        (3, 2, 256,  8, 2),
    ]

    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)

        self.patch_embs = []
        self.enc_blocks = []
        for i, (ps, st, edim, nh, nb) in enumerate(self.STAGE_CFG):
            self.patch_embs.append(PatchEmbedding(ps, st, edim, name=f"pe_{i}"))
            self.enc_blocks.append([
                TransformerEncoderLayer(edim, nh, name=f"enc_{i}_{j}")
                for j in range(nb)
            ])

        target_ch = 128
        self.dec_projs = [
            layers.Conv2D(target_ch, 1, padding="same", name=f"dp_{i}")
            for i in range(len(self.STAGE_CFG))
        ]
        self.dec_ups = [
            layers.UpSampling2D((2**i, 2**i), interpolation="bilinear") if 2**i > 1 else None
            for i in range(len(self.STAGE_CFG))
        ]

        self.dec_conv = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.up1      = layers.UpSampling2D((2,2), interpolation="bilinear")
        self.conv1    = layers.Conv2D(64,  3, padding="same", activation="relu")
        self.up2      = layers.UpSampling2D((2,2), interpolation="bilinear")
        self.conv2    = layers.Conv2D(32,  3, padding="same", activation="relu")
        self.seg_head = layers.Conv2D(num_classes, 1, activation="softmax")

    def call(self, x, training=False):
        feature_maps = []
        for patch_emb, blocks in zip(self.patch_embs, self.enc_blocks):
            tokens, H, W = patch_emb(x)
            for blk in blocks:
                tokens = blk(tokens, H, W, training=training)
            B    = tf.shape(tokens)[0]
            feat = tf.reshape(tokens, (B, H, W, tf.shape(tokens)[-1]))
            feature_maps.append(feat)
            x = feat

        fused = None
        for i, feat in enumerate(feature_maps):
            y = self.dec_projs[i](feat)
            if self.dec_ups[i] is not None:
                y = self.dec_ups[i](y)
            fused = y if fused is None else fused + y

        x = self.dec_conv(fused)
        x = self.up1(x);  x = self.conv1(x)
        x = self.up2(x);  x = self.conv2(x)
        return self.seg_head(x)

