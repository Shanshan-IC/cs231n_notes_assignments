import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    可选择哪种计算距离矩阵的方法进行预测
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    计算test数据到所有training数据的距离，
    返回矩阵dist, dist[i][j]表示测试点i到训练点j的距离
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    for i in range(num_test):
      for j in range(num_train):
        dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
    return dists

  def compute_distances_one_loop(self, X):
    """
    一层循环使用的是numpy的broadcast功能，
    对于每一个test样本（一维），与train计算distance
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis = 1))
    return dists

  def compute_distances_no_loops(self, X):
    """
    不用循环的方法是：
    将平方和展开变成n^2+m^2-2*m*n
    """
    dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis = 1) + np.transpose([np.sum(np.square(X), axis = 1)]))
    return dists

  def predict_labels(self, dists, k=1):
    """
    给定的距离矩阵下，返回预测的label
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      closest_y = self.y_train[np.argsort(dists[i])[:k]] # 前k个label
      y_pred[i] = np.argmax(np.bincount(closest_y)) # 返回样本中出现次数最多
    return y_pred

