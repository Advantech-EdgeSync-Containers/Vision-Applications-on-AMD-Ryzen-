# test_npu.py
import onnxruntime as ort
import numpy as np
import time

def test_npu_basic():
    print("=== AMD NPU 基础测试 ===")
    
    # 1. 检查可用的provider
    providers = ort.get_available_providers()
    print(f"可用Provider: {providers}")
    
    # 2. 创建一个简单的模型用于测试
    if 'VitisAIExecutionProvider' in providers:
        print("\n尝试使用VitisAI...")
        
        # 创建一个极简的ONNX模型（或使用自带的测试模型）
        # 这里假设有一个测试模型在 test_model.onnx
        
        # 使用最简配置
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # 减少日志
        
        try:
            # 尝试不同的目标
            targets = ['RyzenAI', 'VAIML', 'CPU']
            
            for target in targets:
                try:
                    print(f"\n尝试目标: {target}")
                    
                    session = ort.InferenceSession(
                        'test_model.onnx',  # 替换为你的模型路径
                        sess_options=sess_options,
                        providers=['VitisAIExecutionProvider'],
                        provider_options=[{'target': target}]
                    )
                    
                    print(f"✅ {target} 成功")
                    break
                    
                except Exception as e:
                    print(f"❌ {target} 失败: {str(e)[:100]}")
                    
        except Exception as e:
            print(f"❌ VitisAI测试失败: {e}")
    
    # 3. 测试CPU回退
    print("\n=== 测试CPU回退 ===")
    try:
        session = ort.InferenceSession(
            'test_model.onnx',  # 替换为你的模型路径
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        print("✅ CPU模式成功")
    except Exception as e:
        print(f"❌ CPU模式也失败: {e}")

if __name__ == "__main__":
    test_npu_basic()