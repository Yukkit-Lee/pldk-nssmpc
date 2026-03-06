# mpc_diagnose.py
import torch
import time
import sys
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor

def run_party(party_id):
    print(f"\n=== Party {party_id} 诊断模式 ===")
    
    try:
        # 1. 创建参与方
        print(f"[Party {party_id}] 1. 创建 Party2PC...")
        party = Party2PC(party_id, SEMI_HONEST)
        print(f"[Party {party_id}]    ✅ Party 创建成功")
        
        # 2. 启动运行时
        print(f"[Party {party_id}] 2. 启动 PartyRuntime...")
        runtime = PartyRuntime(party)
        runtime.__enter__()
        print(f"[Party {party_id}]    ✅ Runtime 启动成功")
        
        # 3. 建立连接
        print(f"[Party {party_id}] 3. 等待连接...")
        party.online()
        print(f"[Party {party_id}]    ✅ 连接成功！")
        
        if party_id == 0:
            # 4. Party 0: 测试通信
            print(f"\n[Party 0] 4. 测试数据共享...")
            
            # 创建一个非常简单的数据
            data = torch.tensor([42.0])
            print(f"[Party 0]    原始数据: {data}")
            
            # 5. 创建 SecretTensor
            print(f"[Party 0] 5. 创建 SecretTensor...")
            secret = SecretTensor(tensor=data)
            print(f"[Party 0]    ✅ SecretTensor 创建成功，类型: {type(secret).__name__}")
            
            # 6. 等待一下
            print(f"[Party 0] 6. 等待 3 秒...")
            time.sleep(3)
            
            # 7. 尝试重建
            print(f"[Party 0] 7. 尝试重建数据...")
            try:
                result = secret.recon().convert_to_real_field()
                print(f"[Party 0]    ✅ 重建成功！结果: {result}")
                
                # 8. 验证
                if torch.allclose(data, result.cpu() if result.is_cuda else result):
                    print(f"[Party 0]    ✅ 验证成功！数据一致")
                else:
                    print(f"[Party 0]    ❌ 验证失败！数据不一致")
            except Exception as e:
                print(f"[Party 0]    ❌ 重建失败: {e}")
                import traceback
                traceback.print_exc()
            
        else:
            # Party 1: 保持在线，打印状态
            print(f"\n[Party 1] 4. 等待 Party 0 的计算...")
            counter = 0
            while True:
                time.sleep(2)
                counter += 1
                print(f"[Party 1]    心跳 #{counter} (运行中...)")
                
                # 每5次心跳检查一次状态
                if counter % 5 == 0:
                    print(f"[Party 1]    仍在等待，已运行 {counter*2} 秒")
        
    except Exception as e:
        print(f"\n[Party {party_id}] ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n[Party {party_id}] 清理资源...")
        try:
            runtime.__exit__(None, None, None)
            print(f"[Party {party_id}]    ✅ 清理完成")
        except:
            pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python mpc_diagnose.py [party_id]")
        sys.exit(1)
    
    party_id = int(sys.argv[1])
    run_party(party_id)