import numpy as np
import torch
from torch.utils.cpp_extension import load
import pytest
import time

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
kvcomp_lib = load(
    name="kvcomp_interface",
    sources=["kvcomp_interface.cu"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-std=c++17"],
)
kvcomp_retrieval = kvcomp_lib.kvcomp_retrieval
kvcomp_topk = kvcomp_lib.kvcomp_topk

class style():
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def print_red(msg):
    print(style.RED + msg + style.RESET)

def print_green(msg):
    print(style.GREEN + msg + style.RESET)

def print_blue(msg):
    print(style.BLUE + msg + style.RESET)

def print_yellow(msg):
    print(style.YELLOW + msg + style.RESET)

@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("num_repre_blocks", [50, 100])
@pytest.mark.parametrize("dim", [576, 1024])
def test_kvcomp_retrieval(batch_size, num_repre_blocks, dim):
    print(f'''TEST kvcomp_retrieval
{' '*4}total number of queries (a.k.a batch_size): {batch_size}
{' '*4}number of key blocks for each request: {num_repre_blocks // batch_size}
{' '*4}dim (num_heads * hidden_size): {dim}\n''')
    N = num_repre_blocks * 2
    block_size = 128
    dtype=torch.uint8
    query_list = []
    for i in range(batch_size):
        query_list.append(torch.randint(0, 256, (dim,), dtype=dtype).cuda())

    repre_cache = torch.randint(0, 256, (N, block_size, dim), dtype=dtype).cuda()

    rng = np.random.default_rng()
    range_n = np.arange(N)
    repre_index = rng.choice(range_n, size=num_repre_blocks, replace=False)
    repre_index = torch.from_numpy(repre_index).to(torch.int32).cuda()
    q_index = torch.randint(0, batch_size, size = [num_repre_blocks], dtype = torch.int32).cuda()

    score = torch.full((num_repre_blocks,), torch.iinfo(torch.int32).max, dtype=torch.int32).cuda()
    score_sorted = torch.full((num_repre_blocks,), torch.iinfo(torch.int32).max, dtype=torch.int32).cuda()
    index = torch.cat([torch.arange(0, num_repre_blocks / batch_size, dtype=torch.int32) for _ in range(batch_size)]).cuda()
    index_sorted = torch.arange(0, num_repre_blocks, dtype=torch.int32).cuda()
    batch_offset = torch.arange(0, num_repre_blocks, num_repre_blocks / batch_size, dtype=torch.int32).cuda()
    batch_offset = torch.cat([batch_offset, torch.tensor([num_repre_blocks], dtype=torch.int32).cuda()])
    workspace = torch.zeros(10000, dtype=torch.int32).cuda()

    Input = kvcomp_lib.RetrievalInputTensor()
    Input.query_list = query_list
    Input.repre_cache = repre_cache
    Input.q_index = q_index
    Input.repre_index = repre_index
    Input.batch_offset = batch_offset
    Input.workspace = workspace

    Output = kvcomp_lib.RetrievalOutputTensor()
    Output.score = score
    Output.score_sorted = score_sorted
    Output.index_ranged = index
    Output.index_sorted = index_sorted

    start = time.perf_counter_ns()
    kvcomp_retrieval(Input, Output)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_green(f"{' '*4}kvcomp_retrieval host API time: {duration/1e6:.3f} ms")

    def efficient_bit_count(tensor):
        tensor = tensor.to(torch.uint8)
        # 将每个 8 位的整数分解成 8 个 bit，通过掩码逐位累加
        # 使用按位与操作逐位计算 1 的个数
        tensor = tensor.unsqueeze(-1)  # [S, block_size, dim, 1]
        bit_count = 0
        for i in range(8):
            bit_count += tensor & (1 << i) > 0  # 按位与，统计每个位置是否为1

        return bit_count.sum(dim=-1)  # 沿最后一维求和，得到每个元素中 1 的个数


    def naive_hash_retrieval():
        # 1. 将 query_list 堆叠成一个 [N, dim] 的张量
        print(f"query.device: {query.device}")
        query = torch.stack(query_list)  # [N, dim]

        # 2. 根据 q_index 从 query 中选择相应的 query_encoder
        query_encoder = query[q_index]  # [S, dim]
        
        # 3. 根据 repre_index 从 repre_cache 中选择相应的 repre_cache_encoder
        repre_cache_encoder = repre_cache[repre_index]  # [S, block_size, dim]
        
        # 4. 对 query_encoder 和 repre_cache_encoder 进行按位异或
        scores = torch.bitwise_xor(query_encoder.unsqueeze(1), repre_cache_encoder)  # [S, block_size, dim]

        # 5. 对每个 uint8 元素统计 1 的个数
        ones_count = efficient_bit_count(scores)  # [S, block_size, dim] -> 计算每个元素的 1 的数量

        # 6. 沿着 dim 维度求和
        scores_flat = ones_count.sum(dim=-1)  # [S, block_size]

        # 7. 在 block_size 维度上选择最小值（Hamming 距离最小的得分）
        hamming_scores_gt = scores_flat.min(dim=-1).values  # [S]

        # 8. Hamming dist 排序
        index_gt = torch.cat([hamming_scores_gt[s:t].argsort(descending=False) for s, t in zip(batch_offset[:-1], batch_offset[1:])])

        return hamming_scores_gt, index_gt


    start = time.perf_counter_ns()
    score_gt, index_gt = naive_hash_retrieval()
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_red(f"{' '*4}naive_hash_retrieval host API time: {duration/1e6:.3f} ms")

    diff = (score.float() - score_gt.float()).abs()
    intersection = index_gt[torch.isin(index_gt, index_sorted)]
    print_blue(f"{' '*4}score diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max)")
    print_blue(f"{' '*4}index diff: {len(index_gt) - len(intersection)}")
    print("")
