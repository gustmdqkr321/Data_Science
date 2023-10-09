
import torch

x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]]) # 자료구조 tensor
y = torch.FloatTensor([[1,2,3], [4,5,6], [7,8,9]])
print("x =", x)
print("y =", y)

print("size : ", x.size()) # 크기와 차원 확인
print("shape:", x.shape)
print("차원(랭크) : ", x.ndimension())

x0 = torch.unsqueeze(x, 0) # 모양 변경, x에 0번째 차원 추가
x1 = torch.unsqueeze(x, 1)
x2 = torch.unsqueeze(x, 2)
print("x0.shape : ", x0.shape)
print("x1.shape : ", x1.shape)
print("x2.shape : ", x2.shape)
print("x0 = ", x0)
print("x1 = ", x1)
print("x2 = ", x2)

x3 = torch.squeeze(torch.squeeze(x0)) # squeeze(x): 텐서 x에서 크기가 1인 차원 제거
print("x3 =", x3)
print("x3.shape =", x3.shape)

x4 = x.view(9) # 텐서 모양 변경 view(x) : x의 모양으로 변환
x5 = x.view(1,3,3)
print("x4 =", x4)
print("x5  ", x5)

x = torch.FloatTensor([[1,2], [3,4], [5,6]]) # xw + b size = 3,2
w = torch.randn(1,2, dtype=torch.float) # 텐서 크기 설정
b = torch.randn(3,1, dtype=torch.float)

result = torch.mm(x, torch.t(w)) + b
print(result)

w = torch.tensor(1.0, requires_grad=True) # w가 1일때
# 기울기 requires_grad값이 설정된 텐서는 자동으로 기울기를 계산함
a = w*3
l = a**2
l.backward()
print('l을 w로 미분한 값', w.grad)