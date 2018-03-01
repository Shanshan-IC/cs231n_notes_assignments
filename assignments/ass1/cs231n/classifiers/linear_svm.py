import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # 初始化权重都为0
  # loss func = sum(0, s_j - s_yj + 1)
  # 计算损失和梯度
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W) # 估计的y等于W*X
    correct_class_score = scores[y[i]]
    for j in range(num_classes): # 对于每一个样本，margin = 非正确分类的score - 该正确分类的score + 1， 1 是delta
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin # 损失函数不断地加上margin
        dW[:,j] += X[i].T # 梯度,对sum(0, s_j - s_yj + 1) 求导
        dW[:,y[i]] += -X[i].T
  loss /= num_train # 取平均
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W) # 加上正则化损失
  dW += reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  https://www.leiphone.com/news/201801/CCT24XbhTR2b6xuv.html
  参考svm markdown的内容
  向量法计算loss
  Loss = sum(L_i,)/N + lamda * sum(W_k)
  L_i = sum(max(y_i - y+1, 0)
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W) # 求得W*X，获得预测的结果
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) # 返回正确预测的分类的score
  margins = np.maximum(0, scores - correct_class_scores +1)
  margins[range(num_train), list(y)] = 0  # 正确分类的没有loss，归0
  loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W) # loss求和
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W
  return loss, dW
