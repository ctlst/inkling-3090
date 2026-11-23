# FP8 KV cache on the RTX 3090

At 196,608 tokens, the eight verification positions shared each tile of cached keys and values. Nsight Compute measured 384.59 MiB of device DRAM reads for 384 MiB of unique K/V in one full-attention kernel invocation. The read count includes ancillary data.

During prefill, eligible 512-token full-attention chunks with at least 65,536 tokens of history stage up to 16,384 older tokens at a time for FlashAttention-2. The combined K/V staging tensor takes up to 64 MiB (2 × 16,384 tokens × 8 KV heads × 128 dimensions × 2 bytes). That figure covers one allocation. [Packaged source patch](../patches/inkling-sm86.patch)
