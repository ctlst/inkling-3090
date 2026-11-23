"""Run inside the built RTX 3090 image, which includes the SM86 kernels."""

import torch
from vllm.models.inkling2.nvidia.ops.align_sm86 import moe_align_block_size
from vllm.models.inkling2.nvidia.ops.kv_sm86 import decode, encode
from vllm.triton_utils import tl, triton


@triton.jit
def convert(inputs, packed, unpacked, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(inputs + i, mask=i < N, other=0.0)
    tl.store(packed + i, encode(x), mask=i < N)
    tl.store(unpacked + i, decode((i % 256).to(tl.uint8)), mask=i < N)


def check_codec():
    # Exhaust BF16 including signed zeros, subnormals, infinities and tie cases.
    x = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).cuda().float()
    encoded = torch.empty_like(x, dtype=torch.uint8)
    decoded = torch.empty_like(x)
    convert[(256,)](x, encoded, decoded, x.numel(), 256)
    cpu = x.cpu()
    keep = ~torch.isnan(cpu)
    expected = cpu.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(encoded.cpu()[keep], expected[keep])
    reference = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(decoded[:256].cpu(), reference, atol=0, rtol=0, equal_nan=True)
    print("E4M3 encode/decode verified", flush=True)


def check_routes():
    for m in (1, 2, 8, 32, 128, 256):
        for block in (8, 16, 48, 64):
            for invalid in (False, True):
                ids = torch.rand(m, 128).topk(8).indices.int()
                if invalid:
                    ids[::2, :] = -1
                sorted_ids, experts, padded = moe_align_block_size(ids.cuda(), block, 128)
                n = padded.item()
                slots = sorted_ids[:n].cpu().view(-1, block)
                exp = experts[:n // block].cpu().repeat_interleave(block)
                valid = slots.flatten() != ids.numel()
                indices = slots.flatten()[valid].long()
                assert torch.equal(indices.sort().values, (ids.flatten() >= 0).nonzero().flatten())
                assert torch.equal(exp[valid], ids.flatten()[indices])
    print("48 routed batches verified", flush=True)


torch.manual_seed(419)
check_codec()
check_routes()
