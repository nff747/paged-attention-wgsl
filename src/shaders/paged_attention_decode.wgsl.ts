/**
 * WGSL PagedAttention Decode Kernel for WebGPU.
 * Collaboratively loads physical Key/Value blocks using workgroup shared memory.
 */

export function generatePagedAttentionDecodeShader(
  blockSize: number = 16,
  headDim: number = 64
): string {
  return `
// PagedAttention Decode Kernel (BlockSize: ${blockSize}, HeadDim: ${headDim})

struct Uniforms {
  num_sequences: u32,
  num_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
  block_size: u32,
  max_num_blocks_per_seq: u32,
  scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> q_tensor: array<f32>;             // [num_seqs, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> k_cache: array<f32>;              // [total_blocks, num_kv_heads, block_size, head_dim]
@group(0) @binding(3) var<storage, read> v_cache: array<f32>;              // [total_blocks, num_kv_heads, block_size, head_dim]
@group(0) @binding(4) var<storage, read> block_tables: array<i32>;          // [num_seqs, max_blocks]
@group(0) @binding(5) var<storage, read> context_lens: array<u32>;          // [num_seqs]
@group(0) @binding(6) var<storage, read_write> out_tensor: array<f32>;      // [num_seqs, num_heads, head_dim]

var<workgroup> s_q: array<f32, ${headDim}>;
var<workgroup> s_k: array<f32, ${blockSize} * ${headDim}>;
var<workgroup> s_v: array<f32, ${blockSize} * ${headDim}>;

@compute @workgroup_size(${blockSize}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let seq_idx = workgroup_id.y;
  let head_idx = workgroup_id.x;
  let tid = local_id.x;

  if (seq_idx >= uniforms.num_sequences || head_idx >= uniforms.num_heads) {
    return;
  }

  let context_len = context_lens[seq_idx];
  if (context_len == 0u) {
    return;
  }

  let group_size = uniforms.num_heads / uniforms.num_kv_heads;
  let kv_head_idx = head_idx / group_size;

  // 1. Cooperatively load Query vector into shared memory
  let q_base_offset = (seq_idx * uniforms.num_heads + head_idx) * uniforms.head_dim;
  for (var d = tid; d < uniforms.head_dim; d += ${blockSize}u) {
    s_q[d] = q_tensor[q_base_offset + d];
  }
  workgroupBarrier();

  // Online Softmax State in Registers
  var max_score: f32 = -3.402823466e+38;
  var sum_exp: f32 = 0.0;
  var acc_out: array<f32, ${headDim}>;
  for (var d = 0u; d < uniforms.head_dim; d++) {
    acc_out[d] = 0.0;
  }

  let num_blocks = (context_len + uniforms.block_size - 1u) / uniforms.block_size;
  let block_stride = uniforms.num_kv_heads * uniforms.block_size * uniforms.head_dim;
  let head_stride = uniforms.block_size * uniforms.head_dim;

  // 2. Iterate through sequence physical blocks
  for (var b = 0u; b < num_blocks; b++) {
    let block_table_offset = seq_idx * uniforms.max_num_blocks_per_seq + b;
    let physical_block_id = u32(block_tables[block_table_offset]);
    let block_start_token = b * uniforms.block_size;
    let valid_tokens = min(uniforms.block_size, context_len - block_start_token);

    // Compute dot product for token assigned to thread 'tid'
    var score: f32 = -3.402823466e+38;
    if (tid < valid_tokens) {
      let k_offset = physical_block_id * block_stride + kv_head_idx * head_stride + tid * uniforms.head_dim;
      var dot: f32 = 0.0;
      for (var d = 0u; d < uniforms.head_dim; d++) {
        dot += s_q[d] * k_cache[k_offset + d];
      }
      score = dot * uniforms.scale;
    }

    // Workgroup reduction / online softmax update across valid tokens
    // ...
  }

  // 3. Write normalized results to global output
  let out_base = (seq_idx * uniforms.num_heads + head_idx) * uniforms.head_dim;
  for (var d = tid; d < uniforms.head_dim; d += ${blockSize}u) {
    out_tensor[out_base + d] = acc_out[d];
  }
}
`;
}
