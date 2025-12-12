import torch
import math
from torch.utils.cpp_extension import load
import time

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="kvcomp_utils",
    sources=["kvcomp_interface.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)
kvcomp_retrieval = lib.kvcomp_retrieval
kvcomp_topk = lib.kvcomp_topk

b = 4
s = 10
dim = 576
block_size = 128
N = 100
query_list = []
for i in range(b):
    query_list.append(torch.randint(0, 256, (dim,), dtype=torch.uint8).cuda())


repre_cache = torch.randint(0, 256, (N, block_size, dim), dtype=torch.uint8).cuda()
repre_table = torch.arange(0, s, dtype = torch.int32).cuda()
q_table = torch.arange(0, s, dtype = torch.int32).cuda()
q_table = q_table % b
score = torch.full((s,), torch.iinfo(torch.int32).max, dtype=torch.int32).cuda()

score_sorted = torch.full((s,), torch.iinfo(torch.int32).max, dtype=torch.int32).cuda()
index = torch.arange(0, s, dtype=torch.int32).cuda()
index_sorted = torch.arange(0, s, dtype=torch.int32).cuda()
offsets = torch.arange(0, s, math.ceil(s / b), dtype=torch.int32).cuda()
offsets = torch.cat([offsets, torch.tensor([s], dtype=torch.int32).cuda()])
workspace = torch.zeros(10000, dtype=torch.int32).cuda()

start = time.time()
kvcomp_retrieval(query_list, repre_cache, q_table, repre_table, score,
                score_sorted, index, index_sorted, offsets, workspace)
print("launch spent: ", time.time() - start)
torch.cuda.synchronize()
elapsed_cuda = time.time() - start
print(f"kvcomp_retrieval time: {elapsed_cuda:.6f} s")

popcount_table = torch.tensor([bin(i).count('1') for i in range(256)], dtype=torch.int32)

def bit_count(x):
    return popcount_table[x]

def naive_hash_retrieval():
    # 1. 将 query_list 堆叠成一个 [N, dim] 的张量
    query = torch.stack(query_list)  # [N, dim]

    # 2. 根据 q_table 从 query 中选择相应的 query_encoder
    query_encoder = query[q_table]  # [S, dim]
    
    # 3. 根据 repre_table 从 repre_cache 中选择相应的 repre_cache_encoder
    repre_cache_encoder = repre_cache[repre_table]  # [S, block_size, dim]
    
    # 4. 对 query_encoder 和 repre_cache_encoder 进行按位异或
    scores = torch.bitwise_xor(query_encoder.unsqueeze(1), repre_cache_encoder)  # [S, block_size, dim]

    # 5. 对每个 uint8 元素应用查表法
    ones_count = torch.zeros_like(scores, dtype=torch.int32)

    # 5.1 使用查表法统计每个 uint8 数字中 1 的个数
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            for k in range(scores.shape[2]):
                ones_count[i, j, k] = popcount_table[scores[i, j, k].item()]

    # 5.2 沿着 dim 维度求和
    scores_flat = torch.sum(ones_count, dim=-1) # [S, block_size]

    # 6. 在 block_size 维度上选择最小值
    hamming_scores_gt = torch.min(scores_flat, dim=-1).values  # [S]
    
    # 7. 返回输出
    return hamming_scores_gt


start = time.time()
score_gt = naive_hash_retrieval()
torch.cuda.synchronize()
elapsed_naive = time.time() - start
print(f"naive_hash_retrieval time: {elapsed_naive:.6f} s")
print("score_gt: ", score_gt)
print("score_kernel: ", score)
print("score_sorted: ", score_sorted)
print("index_sorted: ", index_sorted)

diff = (score.float() - score_gt.float()).abs()
print("diff: ", diff.mean(), diff.max())

batch_size = 10
total_seq_len = 3200 // 128 * batch_size
topk = 10
num_layers = 61
warmup_iters = 10

score = torch.randint(0, 1000000, (total_seq_len,), dtype=torch.int32).cuda()
index = torch.arange(0, total_seq_len, dtype=torch.int32).cuda()
offsets = torch.arange(0, total_seq_len, math.ceil(total_seq_len / batch_size), dtype=torch.int32).cuda()
offsets = torch.cat([offsets, torch.tensor([total_seq_len], dtype=torch.int32).cuda()])


print(f'''info:
==========
Select {topk} from {total_seq_len//batch_size}
batch_size:{batch_size}, num_layers: {num_layers}
==========''')
print("offsets: ", offsets, offsets.shape)

batch_size = offsets.shape[0] - 1
score_out = torch.zeros(total_seq_len, dtype=torch.int32).cuda()
index_out = torch.zeros(total_seq_len, dtype=torch.int32).cuda()


cost_time = []
workspace = torch.zeros(10000, dtype=torch.int32).cuda()
for iter in range(warmup_iters + num_layers):
    begin = time.time()
    # reset index tensor for radixSort
    # for i in range(total_seq_len):
    #     index[i] = i
    # for i in range(batch_size + 1):
    #     offsets[i] = i * math.ceil(total_seq_len / batch_size)
    kvcomp_topk(score, index, offsets, score_out, index_out, workspace)

    torch.cuda.synchronize()
    duration = time.time() - begin
    if iter >= warmup_iters:
        cost_time.append(duration)

print(f"kvcomp topk: each_layer: {sum(cost_time) / len(cost_time)}, all_layers: {sum(cost_time)}")


cost_time_2 = []
for iter in range(warmup_iters + num_layers):
    begin = time.time()
    gt_index = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        sorted, indices = torch.topk(score[start:stop], dim=0, k=topk, largest=False)
        indices += start
        gt_index.append(indices)
    gt_index = torch.stack(gt_index)
    # score = score.view(batch_size, -1)
    # _, gt_index = torch.topk(score, dim=1, k=topk)
    # gt_index += torch.arange(0, batch_size)[:, None].cuda() * math.ceil(total_seq_len / batch_size)
    gt_index = gt_index.view(-1)
    torch.cuda.synchronize()
    duration = time.time() - begin
    if iter >= warmup_iters:
        cost_time_2.append(duration)

print(f"torch topk: each_layer: {sum(cost_time_2) / len(cost_time_2)}, all_layers: {sum(cost_time_2)}")

index_out = index_out.view(batch_size, -1)
gt_index = gt_index.view(batch_size, -1)
print(index_out.shape, gt_index.shape)
diff = (index_out[:, :topk] - gt_index).abs()
print("diff: ", diff.max())
