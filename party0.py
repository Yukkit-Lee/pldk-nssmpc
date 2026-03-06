# 验证RingTensor基础运算
python -c "from NssMPC.common.ring.ring_tensor import RingTensor; import torch; a=RingTensor(torch.tensor([1,2,3]), ring=2**32); print(a + a)"
 
# 验证秘密共享
python -c "from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing; import torch; data=torch.tensor([10.5]); shares=ArithmeticSecretSharing.share(data, 2); print('Share 0:', shares[0], 'Share 1:', shares[1])"