#!/bin/bash

set -e

echo "========================================"
echo "PEAFOWL 项目环境初始化脚本"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo ""
echo "步骤 1: 检测虚拟环境..."
if [ -d "$VENV_DIR" ]; then
    echo "✓ 虚拟环境已存在"
else
    echo "✗ 虚拟环境不存在，正在创建..."
    python3 -m venv "$VENV_DIR"
    echo "✓ 虚拟环境创建成功"
fi

echo ""
echo "步骤 2: 激活虚拟环境并安装依赖..."
source "${VENV_DIR}/bin/activate"

echo "正在安装依赖包（使用清华源）..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "步骤 3: 运行测试..."
cd "${SCRIPT_DIR}"
set +e
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" pytest "tests/" -v
TEST_RESULT=$?
set -e

if [ $TEST_RESULT -eq 0 ] || [ $TEST_RESULT -eq 1 ]; then
    echo ""
    echo "========================================"
    if [ $TEST_RESULT -eq 0 ]; then
        echo "✓ 所有测试通过！"
    else
        echo "⚠ 测试完成（有错误，但继续启动）"
    fi
    echo "========================================"
    echo ""
    echo "步骤 4: 启动 WebApp..."
    cd "${SCRIPT_DIR}/webapp"
    python app.py
else
    echo ""
    echo "========================================"
    echo "✗ 测试失败，终止启动"
    echo "========================================"
    exit 1
fi
