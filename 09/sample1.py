'''
Created on 2020年9月17日

@author: tony
'''
from collections import Counter
import math

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.models import Sequential, load_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataLoader():

  def __init__(self):
    plt.style.use('default')
    self.df = pd.read_csv('3d_lottery.csv')
    print(self.df.head())
    min_date = min(self.df['开奖日期'])
    max_date = max(self.df['开奖日期'])
    count = len(self.df)
    print('全量数据集：从 {} 到 {} 期间，共计 {} 条。'.format(min_date, max_date, count))

  def trans_num(self):
    self.num_str_list = [[num if len(num) > 0 else '' for num in num_str.split(' ')] for num_str in self.df['中奖号码']]
    print(self.num_str_list[:5])
    num_int_list = [int(''.join(num_str)) for num_str in self.num_str_list]
    print(num_int_list[:5])
    self.df['中奖号码'] = num_int_list
    print(self.df.head())

  def draw_line_chart(self, begin_date, end_date, x, y, title):
    mask = (self.df['开奖日期'] >= begin_date) & (self.df['开奖日期'] <= end_date)
    print(mask)
    year_df = self.df.loc[mask]
    print(year_df)
    plt.figure(figsize = (14, 5))
    plt.plot(year_df[x], year_df[y])
    plt.title(title)
    plt.xlabel('Number of Periods')
    plt.ylabel('Won Numbers')
    plt.grid()
    plt.show()

  def count(self):
    counter = Counter(self.df['中奖号码'])
    print('数字组合共有 {} 种，Top15是 {}。'.format(len(counter.most_common()), counter.most_common(15)))
    other_num_int_list = []
    for i in range(0, 1000):
      found = False
      for num_count in counter.most_common():
        if i == num_count[0]:
          found = True
          break
      if not found:
        other_num_int_list.append(i)
    other_num_str_list = []
    for num_int in other_num_int_list:
      if len(str(num_int)) == 1:
        other_num_str_list.append('00' + str(num_int))
      elif len(str(num_int)) == 2:
        other_num_str_list.append('0' + str(num_int))
      else:
        other_num_str_list.append(str(num_int))
    print('未出现的数字组合是 {}。'.format(other_num_str_list))

  def split_sequence(self, seq, step):
    x, y = list(), list()
    sequence = seq[::-1]
    size = len(sequence)
    for i in range(size):
      idx = i + step
      if idx > (size - 1):
        break
      seq_x, seq_y = sequence[i:idx], sequence[size - 1 - idx]
      x.append(seq_x)
      y.append(seq_y)
    return np.array(x), np.array(y)

  def train_my_model(self, x, y, model_path):
    x = x.reshape(x.shape[0], x.shape[1], 1)
    lstm_predictor_model = LSTM1(x, y, None, None).train_lstm_predictor(True)
    lstm_predictor_model.save(model_path)


class LSTM1():

  def __init__(self, x1, y1, x2, y2):
    self.x_train = x1
    self.y_train = y1
    self.x_test = x2
    self.y_test = y2

  def predict_by_DecisionTreeClassifier(self):
    model = DecisionTreeClassifier()
    model.fit(self.x_train, self.y_train)
    train_predict = model.predict(self.x_train)
    test_predict = model.predict(self.x_test)
    train_accuracy = accuracy_score(self.y_train, train_predict)
    test_accuracy = accuracy_score(self.y_test, test_predict)
    print('决策树分类器的训练精度：{}.'.format(train_accuracy))
    print('决策树分类器的测试精度：{}.'.format(test_accuracy))

  def predict_by_MLPClassifier(self):
    model = MLPClassifier(hidden_layer_sizes = 128, batch_size = 64, max_iter = 1000, solver = 'adam')
    model.fit(self.x_train, self.y_train)
    train_predict = model.predict(self.x_train)
    test_predict = model.predict(self.x_test)
    train_accuracy = accuracy_score(self.y_train, train_predict)
    test_accuracy = accuracy_score(self.y_test, test_predict)
    print('多层感知器分类器的训练精度：{}.'.format(train_accuracy))
    print('多层感知器分类器的测试精度：{}.'.format(test_accuracy))

  def train_lstm_predictor(self, need_clear_session = False):
    if need_clear_session:
      backend.clear_session()
    model = Sequential()
    model.add(LSTM(50, activation = 'relu', input_shape = (3, 1)))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer = 'adam', loss = 'mse')
    x, y = np.array(self.x_train), np.array(self.y_train)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    model.fit(x, y, epochs = 1000, verbose = 0)
    return model


class MLP():

  def __init__(self, x, y):
    self.x_train = x
    self.y_train = y

  def train_mlp_predictor(self, need_clear_session = False):
    if need_clear_session:
      backend.clear_session()
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_dim = 3))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mse')
    model.summary()
    x, y = np.array(self.x_train), np.array(self.y_train)
    model.fit(x, y, epochs = 2000, verbose = 0)
    return model


class CNN():

  def __init__(self, x, y):
    self.x_train = x
    self.y_train = y

  def train_cnn_predictor(self, need_clear_session = False):
    if need_clear_session:
      backend.clear_session()
    model = Sequential()
    model.add(Conv1D(filters = 64, kernel_size = 2, activation = 'relu', input_shape = (3, 1)))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mse')
    model.summary()
    x, y = np.array(self.x_train), np.array(self.y_train)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    model.fit(x, y, epochs = 1000, verbose = 0)
    return model


if __name__ == '__main__':
  start_date = '2018-01-01'
  end_date = '2018-12-31'
  data_loader = DataLoader()
  data_loader.trans_num()
  data_loader.draw_line_chart(start_date, end_date, '期号', '中奖号码', 'Won Numbers of Periods, Date from {} to {}'.format(start_date, end_date))
  data_loader.count()
  x, y = data_loader.split_sequence(data_loader.df['中奖号码'], 3)
  test_ratio = 0.15
  feature_len = len(x)
  test_feature_len = int(feature_len * test_ratio)
  # 这里 sklearn.model_selection 的 train_test_split() 方法不能用，需要保证顺序
  x_train, y_train = x[test_feature_len:], y[test_feature_len:]
  x_test, y_test = x[:test_feature_len], y[:test_feature_len]
  print('x_train.shape={}, y_train.shape={}。'.format(x_train.shape, y_train.shape))
  print('x_test.shape={}, y_test.shape={}。'.format(x_test.shape, y_test.shape))
  _x = x[len(x) - 10:]
  _y = y[len(x) - 10:]
  for i, v in enumerate(_x):
    print(v, _y[i])
  print('\n************************************************************\n')

  model1 = LSTM1(x_train, y_train, x_test, y_test)
  model1.predict_by_DecisionTreeClassifier()
  model1.predict_by_MLPClassifier()
  print('\n************************************************************\n')

  model2 = MLP(x_train, y_train)
  mlp_predictor_model = model2.train_mlp_predictor(True)
  _x_test = np.array(x_test[0]).reshape(1, 3)
  y_hat = mlp_predictor_model.predict(_x_test, verbose = 0)
  print('多层感知器的预测值：{}。'.format(y_hat))
  y_hat = mlp_predictor_model.predict(x_test, verbose = 0)
  for i in range(10):
    print('真实值={} : {}=预测值。'.format(y_test[i], math.ceil(y_hat[i])))
  print('\n************************************************************\n')

  model3 = CNN(x_train, y_train)
  cnn_predictor_model = model3.train_cnn_predictor(True)
  _x_test = np.array(x_test[0]).reshape(1, 3, 1)
  y_hat = cnn_predictor_model.predict(_x_test, verbose = 0)
  print('卷积神经网络的预测值：{}。'.format(y_hat))
  print('\n************************************************************\n')

  model4 = LSTM1(x_train, y_train, x_test, y_test)
  lstm_predictor_model = model4.train_lstm_predictor(True)
  _x_test = np.array(x_test[0]).reshape(1, 3, 1)
  y_hat = lstm_predictor_model.predict(_x_test, verbose = 0)
  print('TF长短期记忆网络的预测值：{}。'.format(y_hat))
  _x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
  y_hat = lstm_predictor_model.predict(_x_test, verbose = 0)
  for i in range(10):
    print('真实值={} : {}=预测值。'.format(y_test[i], math.ceil(y_hat[i])))
  print('\n************************************************************\n')

  num_str_list_2 = [[int(n) for n in t] for t in data_loader.num_str_list]
  first_column = np.array(num_str_list_2)[:, 0]
  second_column = np.array(num_str_list_2)[:, 1]
  third_column = np.array(num_str_list_2)[:, 2]
  first_x, first_y = data_loader.split_sequence(first_column, 3)
  second_x, second_y = data_loader.split_sequence(second_column, 3)
  third_x, third_y = data_loader.split_sequence(third_column, 3)
  first_model_path = 'first_model.h5'
  second_model_path = 'second_model.h5'
  third_model_path = 'third_model.h5'
  data_loader.train_my_model(first_x, first_y, first_model_path)
  data_loader.train_my_model(second_x, second_y, second_model_path)
  data_loader.train_my_model(third_x, third_y, third_model_path)
  first_model = load_model(first_model_path)
  second_model = load_model(second_model_path)
  third_model = load_model(third_model_path)
  first_test_x, first_test_y = data_loader.split_sequence(first_column[:10], 3)
  second_test_x, second_test_y = data_loader.split_sequence(second_column[:10], 3)
  third_test_x, third_test_y = data_loader.split_sequence(third_column[:10], 3)
  first_test_x = first_test_x.reshape(first_test_x.shape[0], first_test_x.shape[1], 1)
  second_test_x = second_test_x.reshape(second_test_x.shape[0], second_test_x.shape[1], 1)
  third_test_x = third_test_x.reshape(third_test_x.shape[0], third_test_x.shape[1], 1)
  first_predicted_value = first_model.predict(first_test_x, verbose = 0)
  second_predicted_value = second_model.predict(second_test_x, verbose = 0)
  third_predicted_value = third_model.predict(third_test_x, verbose = 0)
  final_predicted_x = np.vstack([first_predicted_value[:, 0], second_predicted_value[:, 0], third_predicted_value[:, 0]])
  final_target_x = np.vstack([first_test_y[:10].tolist(), second_test_y[:10].tolist(), third_test_y[:10].tolist()])
  final_predicted_x = final_predicted_x.transpose()
  final_target_x = final_target_x.transpose()
  for i, v in enumerate(final_predicted_x):
    print('P：{} vs T：{}。'.format([math.ceil(p) for p in final_predicted_x[i]], final_target_x[i]))
  print('\n************************************************************\n')
