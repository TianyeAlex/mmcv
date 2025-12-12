#!/bin/bash
set -e

echo "======================================"
echo "🔧 在 Docker 中重新编译原版 MMCV"
echo "======================================"

# Check if Docker container is running
if ! docker ps | grep -q "uniad2.0"; then
    echo "❌ Docker container 'uniad2.0' is not running"
    exit 1
fi

echo "✅ Docker 容器 uniad2.0 正在运行"

# Run compilation in Docker
docker exec uniad2.0 bash -c "
set -e
source /root/workspace/test/env.sh
cd /root/workspace/test/mmcv

echo '======================================'
echo '📦 步骤 1/3: 清理之前的编译产物'
echo '======================================'
rm -rf build/
rm -f mmcv/_ext.*.so
rm -f mmcv/_ext_opt.*.so

echo '======================================'
echo '📦 步骤 2/3: 重新编译原版 MMCV (包含 Stage 1 L1 cache 优化)'
echo '======================================'
python setup.py build_ext --inplace
python setup_opt.py build_ext --inplace
python test_deform_attn.py"
