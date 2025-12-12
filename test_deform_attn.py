"""
Realistic performance test for MultiScaleDeformableAttnFunction in mmcv
based on actual UniAD stage1_track_map configuration.

完整测试 UniAD 中 Deformable Attention 的三个模块:
1. Temporal Self-Attention (BEV 自注意力)
2. Spatial Cross-Attention (BEV ↔ 多尺度图像特征)
3. Detection Decoder (物体查询 → BEV)

每个模块模拟完整的前向+反向传播，准确反映实际训练场景。
每个模块重复测试10次取平均值，减少噪声误差。

USAGE:
  python test_uniad_full.py
"""
import torch
import time
import numpy as np
import sys
import os

# Import from local mmcv
mmcv_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mmcv_path)

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

# Try to import optimized version
try:
    from mmcv.ops.multi_scale_deform_attn_opt import MultiScaleDeformableAttnFunctionOpt
    OPTIMIZED_AVAILABLE = True
    MultiScaleDeformableAttnFunction_Optimized = MultiScaleDeformableAttnFunctionOpt
except ImportError:
    OPTIMIZED_AVAILABLE = False
    MultiScaleDeformableAttnFunction_Optimized = None


def create_test_configs():
    """
    创建三个模块的测试配置，基于 UniAD Stage1 Track Map 的实际参数
    
    UniAD 架构:
    - Encoder: 6 layers, 每层包含 Temporal Self-Attention + Spatial Cross-Attention
    - Decoder: 6 layers, 每层包含 Detection Decoder (Self-Attention 使用标准 Attention)
    
    Image size: 900H × 1600W
    BEV size: 200 × 200
    ResNet backbone outputs C3, C4, C5 (stride 8, 16, 32)
    FPN outputs 4 levels: P3, P4, P5, P6 (stride 8, 16, 32, 64)
    """
    device = 'cuda'
    
    # 通用参数
    bev_h, bev_w = 200, 200
    embed_dims = 256
    num_heads = 8
    embed_dim_per_head = embed_dims // num_heads  # 32
    
    # Spatial shapes for multi-level features (FPN outputs) - 基于实际图像尺寸
    # Input image: 900H × 1600W
    spatial_shapes_spatial = torch.tensor([
        [116, 200],  # Level 0 (P3): stride=8  → 928/8 × 1600/8 ≈ 116 × 200
        [58, 100],   # Level 1 (P4): stride=16 → 928/16 × 1600/16 = 58 × 100
        [29, 50],    # Level 2 (P5): stride=32 → 928/32 × 1600/32 ≈ 29 × 50
        [15, 25],    # Level 3 (P6): stride=64 → 960/64 × 1600/64 = 15 × 25
    ], dtype=torch.long, device=device)
    
    # Calculate total keys and level start indices
    num_keys_spatial = sum([h * w for h, w in spatial_shapes_spatial.tolist()])  # 23200 + 5800 + 1450 + 375 = 30825
    level_start_index_spatial = torch.cat([
        torch.tensor([0], dtype=torch.long, device=device),
        spatial_shapes_spatial.prod(1).cumsum(0)[:-1]
    ])
    
    configs = {
        'Temporal Self-Attention': {
            'description': 'BEV feature (200×200) self-attention, 1 level, 4 points',
            'details': f'Keys: {bev_h * bev_w:,}, Queries: {bev_h * bev_w:,}, Levels: 1, Points: 4',
            'num_keys': bev_h * bev_w,  # 40,000
            'num_queries': bev_h * bev_w,  # 40,000
            'num_levels': 1,
            'num_points': 4,
            'spatial_shapes': torch.tensor([[bev_h, bev_w]], dtype=torch.long, device=device),
            'level_start_index': torch.tensor([0], dtype=torch.long, device=device),
            'num_heads': num_heads,
            'embed_dim_per_head': embed_dim_per_head,
        },
        'Spatial Cross-Attention': {
            'description': 'BEV (200×200) cross-attend to multi-scale features, 4 levels, 8 points',
            'details': f'Keys: {num_keys_spatial:,}, Queries: {bev_h * bev_w:,}, Levels: 4, Points: 8',
            'num_keys': num_keys_spatial,
            'num_queries': bev_h * bev_w,  # 40,000
            'num_levels': 4,
            'num_points': 8,
            'spatial_shapes': spatial_shapes_spatial,
            'level_start_index': level_start_index_spatial,
            'num_heads': num_heads,
            'embed_dim_per_head': embed_dim_per_head,
        },
        'Detection Decoder': {
            'description': '900 object queries attend to BEV (200×200), 1 level, 4 points',
            'details': f'Keys: {bev_h * bev_w:,}, Queries: 900, Levels: 1, Points: 4',
            'num_keys': bev_h * bev_w,  # 40,000
            'num_queries': 900,
            'num_levels': 1,
            'num_points': 4,
            'spatial_shapes': torch.tensor([[bev_h, bev_w]], dtype=torch.long, device=device),
            'level_start_index': torch.tensor([0], dtype=torch.long, device=device),
            'num_heads': num_heads,
            'embed_dim_per_head': embed_dim_per_head,
        }
    }
    
    return configs


def benchmark_single_module(config, attn_func, dtype, num_warmup=20, num_iter=100):
    """
    测试单个模块的性能
    
    Args:
        config: 模块配置
        attn_func: 使用的 attention 函数 (原版或优化版)
        dtype: 数据类型 (torch.float32 或 torch.bfloat16)
        num_warmup: 预热迭代次数
        num_iter: 测试迭代次数
    """
    device = 'cuda'
    batch_size = 1
    
    num_keys = config['num_keys']
    num_queries = config['num_queries']
    num_levels = config['num_levels']
    num_points = config['num_points']
    spatial_shapes = config['spatial_shapes']
    level_start_index = config['level_start_index']
    num_heads = config['num_heads']
    embed_dim_per_head = config['embed_dim_per_head']
    
    im2col_step = torch.tensor(64, dtype=torch.long, device=device)
    
    # ====== 预热阶段 ======
    for _ in range(num_warmup):
        # 生成输入数据
        value = torch.randn(
            batch_size, num_keys, num_heads, embed_dim_per_head,
            dtype=dtype, device=device, requires_grad=True
        )
        sampling_locations = torch.rand(
            batch_size, num_queries, num_heads, num_levels, num_points, 2,
            dtype=dtype, device=device, requires_grad=True
        )
        attention_weights = torch.rand(
            batch_size, num_queries, num_heads, num_levels, num_points,
            dtype=dtype, device=device, requires_grad=True
        )
        # 归一化 attention weights (模拟 softmax 输出)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        # Forward
        output = attn_func.apply(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, im2col_step
        )
        
        # Backward (模拟训练场景)
        loss = output.sum()
        loss.backward()
        
        # 清理
        del value, sampling_locations, attention_weights, output, loss
    
    torch.cuda.synchronize()
    
    # ====== 测试阶段 ======
    times_fwd = []
    times_bwd = []
    times_total = []
    
    for _ in range(num_iter):
        # 生成输入数据
        value = torch.randn(
            batch_size, num_keys, num_heads, embed_dim_per_head,
            dtype=dtype, device=device, requires_grad=True
        )
        sampling_locations = torch.rand(
            batch_size, num_queries, num_heads, num_levels, num_points, 2,
            dtype=dtype, device=device, requires_grad=True
        )
        attention_weights = torch.rand(
            batch_size, num_queries, num_heads, num_levels, num_points,
            dtype=dtype, device=device, requires_grad=True
        )
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        torch.cuda.synchronize()
        start_total = time.time()
        
        # ====== Forward Pass ======
        start_fwd = time.time()
        output = attn_func.apply(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, im2col_step
        )
        torch.cuda.synchronize()
        time_fwd = time.time() - start_fwd
        times_fwd.append(time_fwd)
        
        # ====== Backward Pass ======
        loss = output.sum()
        start_bwd = time.time()
        loss.backward()
        torch.cuda.synchronize()
        time_bwd = time.time() - start_bwd
        times_bwd.append(time_bwd)
        
        time_total = time.time() - start_total
        times_total.append(time_total)
        
        # 清理
        del value, sampling_locations, attention_weights, output, loss
    
    # 计算统计信息
    results = {
        'forward_ms': np.mean(times_fwd) * 1000,
        'forward_std': np.std(times_fwd) * 1000,
        'backward_ms': np.mean(times_bwd) * 1000,
        'backward_std': np.std(times_bwd) * 1000,
        'total_ms': np.mean(times_total) * 1000,
        'total_std': np.std(times_total) * 1000,
    }
    
    return results


def test_all_modules(dtype_name='FP32', use_optimized=False, num_iter=100, num_repeats=10):
    """
    测试所有三个模块，每个模块重复测试多次取平均值以减少噪声
    
    Args:
        dtype_name: 'FP32' 或 'BF16'
        use_optimized: 是否使用优化版本
        num_iter: 每次测试的迭代次数
        num_repeats: 每个模块重复测试的次数（取平均值）
    """
    dtype = torch.float32 if dtype_name == 'FP32' else torch.bfloat16
    
    if use_optimized and OPTIMIZED_AVAILABLE:
        attn_func = MultiScaleDeformableAttnFunction_Optimized
        version_name = f"{dtype_name}-Optimized"
    else:
        attn_func = MultiScaleDeformableAttnFunction
        version_name = f"{dtype_name}-Original"
    
    configs = create_test_configs()
    
    print(f"\n{'='*100}")
    print(f"Testing: {version_name}")
    print(f"{'='*100}")
    
    all_results = {}
    
    for module_name, config in configs.items():
        print(f"\n📊 Module: {module_name}")
        print(f"   {config['description']}")
        print(f"   {config['details']}")
        
        # 收集多次重复测试的结果
        repeat_results = []
        
        print(f"   Warmup (20 iterations)...", end='', flush=True)
        _ = benchmark_single_module(config, attn_func, dtype, num_warmup=20, num_iter=num_iter)
        print(f" Done")
        
        print(f"   Testing ({num_iter} iterations)...", end='', flush=True)
        for repeat_idx in range(num_repeats):
            result = benchmark_single_module(config, attn_func, dtype, num_warmup=0, num_iter=num_iter)
            repeat_results.append(result)
            if (repeat_idx + 1) % 2 == 0:
                print(f" {repeat_idx + 1}/{num_repeats}", end='', flush=True)
        print(f" Done")
        
        # 计算所有重复测试的平均值
        avg_results = {
            'forward_ms': np.mean([r['forward_ms'] for r in repeat_results]),
            'forward_std': np.mean([r['forward_std'] for r in repeat_results]),
            'backward_ms': np.mean([r['backward_ms'] for r in repeat_results]),
            'backward_std': np.mean([r['backward_std'] for r in repeat_results]),
            'total_ms': np.mean([r['total_ms'] for r in repeat_results]),
            'total_std': np.mean([r['total_std'] for r in repeat_results]),
        }
        
        all_results[module_name] = avg_results
        
        print(f"   Results:")
        print(f"     Forward:    {avg_results['forward_ms']:5.2f} ± {avg_results['forward_std']:4.2f} ms")
        print(f"     Backward:   {avg_results['backward_ms']:5.2f} ± {avg_results['backward_std']:4.2f} ms")
        print(f"     Total:      {avg_results['total_ms']:5.2f} ± {avg_results['total_std']:4.2f} ms")
    
    return version_name, all_results


def estimate_full_pipeline_time(results_dict):
    """
    估算完整 UniAD 流程中 Deformable Attention 的总耗时
    
    UniAD Stage1 架构:
    - Encoder: 6 layers
      - 每层: Temporal Self-Attention + Spatial Cross-Attention
    - Decoder: 6 layers
      - 每层: Detection Decoder (Cross-Attention)
    """
    temporal = results_dict['Temporal Self-Attention']['total_ms']
    spatial = results_dict['Spatial Cross-Attention']['total_ms']
    decoder = results_dict['Detection Decoder']['total_ms']
    
    # Encoder: 6 layers × (Temporal + Spatial)
    encoder_time = 6 * (temporal + spatial)
    
    # Decoder: 6 layers × Decoder
    decoder_time = 6 * decoder
    
    # Total
    total_time = encoder_time + decoder_time
    
    return {
        'encoder': encoder_time,
        'decoder': decoder_time,
        'total': total_time,
        'per_module': {
            'temporal': temporal,
            'spatial': spatial,
            'detection': decoder
        }
    }


def main():
    print("="*100)
    print("UniAD Deformable Attention 完整性能测试 (FP32)")
    print("="*100)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if OPTIMIZED_AVAILABLE:
        print(f"✅ Optimized version: Available")
    else:
        print(f"⚠️  Optimized version: Not available")
    
    print(f"\n测试配置:")
    print(f"  - 每个模块迭代次数: 100")
    print(f"  - 每个模块重复测试: 10次 (取平均值)")
    print(f"  - 预热次数: 20")
    print(f"  - 模拟完整前向+反向传播")
    
    # ====== 测试 FP32 原版 ======
    print(f"\n{'🔵 '*50}")
    version_fp32_orig, results_fp32_orig = test_all_modules('FP32', use_optimized=False, num_iter=100, num_repeats=10)
    
    # ====== 测试 FP32 优化版 ======
    if OPTIMIZED_AVAILABLE:
        print(f"\n{'🟢 '*50}")
        version_fp32_opt, results_fp32_opt = test_all_modules('FP32', use_optimized=True, num_iter=100, num_repeats=10)
    
    # ====== 综合对比 ======
    print(f"\n{'='*100}")
    print("📊 性能对比总结")
    print(f"{'='*100}")
    
    modules = ['Temporal Self-Attention', 'Spatial Cross-Attention', 'Detection Decoder']
    
    for module in modules:
        print(f"\n{module}:")
        print(f"  {'Version':<20} {'Forward':<18} {'Backward':<18} {'Total':<18}")
        print(f"  {'-'*74}")
        
        # FP32 Original
        r = results_fp32_orig[module]
        print(f"  {'FP32-Original':<20} {r['forward_ms']:6.2f}±{r['forward_std']:4.2f} ms   "
              f"{r['backward_ms']:6.2f}±{r['backward_std']:4.2f} ms   "
              f"{r['total_ms']:6.2f}±{r['total_std']:4.2f} ms")
        
        # FP32 Optimized
        if OPTIMIZED_AVAILABLE:
            r = results_fp32_opt[module]
            speedup_fp32 = results_fp32_orig[module]['total_ms'] / r['total_ms']
            print(f"  {'FP32-Optimized':<20} {r['forward_ms']:6.2f}±{r['forward_std']:4.2f} ms   "
                  f"{r['backward_ms']:6.2f}±{r['backward_std']:4.2f} ms   "
                  f"{r['total_ms']:6.2f}±{r['total_std']:4.2f} ms   "
                  f"({speedup_fp32:.3f}x)")
    
    # ====== 完整流程估算 ======
    print(f"\n{'='*100}")
    print("🎯 完整 UniAD 流程估算 (Encoder 6层 + Decoder 6层)")
    print(f"{'='*100}")
    
    pipeline_fp32_orig = estimate_full_pipeline_time(results_fp32_orig)
    
    print(f"\n{'Version':<20} {'Encoder (6×2)':<15} {'Decoder (6×1)':<15} {'Total':<15}")
    print(f"{'-'*65}")
    print(f"{'FP32-Original':<20} {pipeline_fp32_orig['encoder']:7.2f} ms    "
          f"{pipeline_fp32_orig['decoder']:7.2f} ms    "
          f"{pipeline_fp32_orig['total']:7.2f} ms")
    
    if OPTIMIZED_AVAILABLE:
        pipeline_fp32_opt = estimate_full_pipeline_time(results_fp32_opt)
        speedup = pipeline_fp32_orig['total'] / pipeline_fp32_opt['total']
        print(f"{'FP32-Optimized':<20} {pipeline_fp32_opt['encoder']:7.2f} ms    "
              f"{pipeline_fp32_opt['decoder']:7.2f} ms    "
              f"{pipeline_fp32_opt['total']:7.2f} ms    "
              f"({speedup:.3f}x)")
        
        # 优化收益总结
        print(f"\n{'='*100}")
        print("✅ 优化收益总结")
        print(f"{'='*100}")
        
        fp32_saved = pipeline_fp32_orig['total'] - pipeline_fp32_opt['total']
        fp32_speedup = pipeline_fp32_orig['total'] / pipeline_fp32_opt['total']
        
        print(f"\nFP32 优化:")
        print(f"  原版: {pipeline_fp32_orig['total']:.2f} ms → 优化版: {pipeline_fp32_opt['total']:.2f} ms")
        print(f"  加速: {fp32_speedup:.3f}x ({(fp32_speedup-1)*100:.1f}% 提升)")
        print(f"  每次迭代节省: {fp32_saved:.2f} ms")
        print(f"\n训练收益 (30K iterations):")
        print(f"  节省时间: {fp32_saved * 30000 / 1000 / 60:.1f} 分钟 ({fp32_saved * 30000 / 1000 / 3600:.1f} 小时)")
    
    print(f"\n{'='*100}")
    print("✅ 测试完成!")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
