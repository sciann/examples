# ==============================================================================
# Copyright 2021 SciANN -- Ehsan Haghighat. 
# All Rights Reserved.
#
# Licensed under the MIT License.
# 
# A guide for generating collocation points for PINN solvers. 
# 
# Includes: 
#    - DataGeneratorX: 
#           Generate 1D collocation grid. 
#    - DataGeneratorXY: 
#           Generate 2D collocation grid for a rectangular domain. 
#    - DataGeneratorXT: 
#           Generate 1D time-dependent collocation grid. 
#    - DataGeneratorXYT: 
#           Generate 2D time-dependent collocation grid  for a rectangular domain. 
# ==============================================================================

import numpy as np 
import matplotlib.pyplot as plt 
from itertools import cycle
cycol = cycle('bgrcmk')


class DataGeneratorX:
  """ Generates 1D collocation grid for training PINNs
  # Arguments:
    X: [X0, X1]
    targets: list and type of targets you wish to impose on PINNs. 
        ('domain', 'bc-left', 'bc-right', 'all')
    num_sample: total number of collocation points. 

  # Examples: 
    >> dg = DataGeneratorX([0., 1.], ["domain", "bc-left", "bc-right"], 10000)
    >> input_data, target_data = dg.get_data()

  """
  def __init__(self, 
               X=[0., 1.],
               targets=['domain', 'bc-left', 'bc-right'], 
               num_sample=10000):
    'Initialization'
    self.Xdomain = X
    self.targets = targets
    self.num_sample = num_sample
    self.input_data = None
    self.target_data = None
    self.set_data()

  def __len__(self):
    return self.input_data[0].shape[0]

  def set_data(self):
    self.input_data, self.target_data = self.generate_data()

  def get_data(self):
    return self.input_data, self.target_data

  def generate_data(self):
    # distribute half inside domain half on the boundary 
    num_sample = int(self.num_sample/2)
    
    counter = 0
    # domain points 
    x_dom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
    ids_dom = np.arange(x_dom.shape[0])
    counter += ids_dom.size

    # left bc points 
    x_bc_left = np.full(int(num_sample/2), self.Xdomain[0])
    ids_bc_left = np.arange(x_bc_left.shape[0]) + counter
    counter += ids_bc_left.size

    # right bc points 
    x_bc_right = np.full(num_sample-int(num_sample/2), self.Xdomain[1])
    ids_bc_right = np.arange(x_bc_right.shape[0]) + counter 
    counter += ids_bc_right.size

    ids_bc = np.concatenate([ids_bc_left, ids_bc_right])
    ids_all = np.concatenate([ids_dom, ids_bc])

    ids = {
        'domain': ids_dom, 
        'bc-left': ids_bc_left, 
        'bc-right': ids_bc_right,
        'bc': ids_bc, 
        'all': ids_all
    }

    assert all([t in ids.keys() for t in self.targets]), \
      'accepted target types: {}'.format(ids.keys())

    input_data = [
        np.concatenate([x_dom, x_bc_left, x_bc_right]).reshape(-1,1),
    ]
    total_sample = input_data[0].shape[0]

    target_data = []
    for i, tp in enumerate(self.targets):
      target_data.append(
          (ids[tp], 'zeros')
      )
      
    return input_data, target_data

  def get_test_grid(self, Nx=1000):
    xs = np.linspace(self.Xdomain[0], self.Xdomain[1], Nx)
    return xs

  def plot_sample_batch(self, batch_size=500):
      ids = np.random.choice(len(self), batch_size, replace=False)
      x_data = self.input_data[0][ids,:]
      y_data = np.random.uniform(-.1, .1, x_data.shape)
      plt.scatter(x_data, y_data)
      plt.xlabel('x')
      plt.ylabel('Random vals')
      plt.ylim(-1,1)
      plt.title('Sample batch = {}'.format(batch_size))
      plt.show()

  def plot_data(self):
      fig = plt.figure()
      for t, (t_idx, t_val) in zip(self.targets, self.target_data):
        x_data = self.input_data[0][t_idx,:]
        y_data = np.random.uniform(-.1, .1, x_data.shape)
        plt.scatter(x_data, y_data, label=t, c=next(cycol))
      plt.ylim(-1,1)
      plt.xlabel('x')
      plt.ylabel('Random vals')
      plt.title('Training Data')
      plt.legend(title="Training Data", bbox_to_anchor=(1.05, 1), loc='upper left')
      fig.tight_layout()
      plt.show()


class DataGeneratorXY:
  """ Generates 2D collocation grid for a rectangular domain
  # Arguments:
    X: [X0, X1]
    Y: [Y0, Y1]
    targets: list and type of targets you wish to impose on PINNs. 
        ('domain', 'bc-left', 'bc-right', 'bc-bot', 'bc-top', 'all')
    num_sample: total number of collocation points. 

  # Examples: 
    >> dg = DataGeneratorXY([0., 1.], [0., 1.], ["domain", "bc-left", "bc-right"], 10000)
    >> input_data, target_data = dg.get_data()

  """
  def __init__(self, 
               X=[0., 1.],
               Y=[0., 1.],
               targets=['domain', 'bc-left', 'bc-right', 'bc-bot', 'bc-top'], 
               num_sample=10000):
    'Initialization'
    self.Xdomain = X
    self.Ydomain = Y
    self.targets = targets
    self.num_sample = num_sample
    self.input_data = None
    self.target_data = None
    self.set_data()

  def __len__(self):
    return self.input_data[0].shape[0]

  def set_data(self):
    self.input_data, self.target_data = self.generate_data()

  def get_data(self):
    return self.input_data, self.target_data

  def generate_data(self):
    # distribute half inside domain half on the boundary 
    num_sample = int(self.num_sample/2)
    
    counter = 0
    # domain points 
    x_dom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
    y_dom = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample)
    ids_dom = np.arange(x_dom.shape[0])
    counter += ids_dom.size

    # bc points 
    num_sample_per_edge = int(num_sample/4)
    # left bc points 
    x_bc_left = np.full(num_sample_per_edge, self.Xdomain[0])
    y_bc_left = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample_per_edge)
    ids_bc_left = np.arange(x_bc_left.shape[0]) + counter
    counter += ids_bc_left.size

    # right bc points 
    x_bc_right = np.full(num_sample_per_edge, self.Xdomain[1])
    y_bc_right = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample_per_edge)
    ids_bc_right = np.arange(x_bc_right.shape[0]) + counter 
    counter += ids_bc_right.size

    # bot bc points 
    x_bc_bot = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample_per_edge)
    y_bc_bot = np.full(num_sample_per_edge, self.Ydomain[0])
    ids_bc_bot = np.arange(x_bc_bot.shape[0]) + counter
    counter += ids_bc_bot.size

    # top bc points 
    x_bc_top = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample_per_edge)
    y_bc_top = np.full(num_sample_per_edge, self.Ydomain[1])
    ids_bc_top = np.arange(x_bc_top.shape[0]) + counter
    counter += ids_bc_top.size

    ids_bc = np.concatenate([ids_bc_left, ids_bc_right, ids_bc_bot, ids_bc_top])
    ids_all = np.concatenate([ids_dom, ids_bc])

    ids = {
        'domain': ids_dom, 
        'bc-left': ids_bc_left, 
        'bc-right': ids_bc_right,
        'bc-bot': ids_bc_bot,
        'bc-top': ids_bc_top,
        'bc': ids_bc,
        'all': ids_all
    }

    assert all([t in ids.keys() for t in self.targets]), \
      'accepted target types: {}'.format(ids.keys())

    input_data = [
        np.concatenate([x_dom, x_bc_left, x_bc_right, x_bc_bot, x_bc_top]).reshape(-1,1),
        np.concatenate([y_dom, y_bc_left, y_bc_right, y_bc_bot, y_bc_top]).reshape(-1,1),
    ]
    total_sample = input_data[0].shape[0]

    target_data = []
    for i, tp in enumerate(self.targets):
      target_data.append(
          (ids[tp], 'zeros')
      )
      
    return input_data, target_data

  def get_test_grid(self, Nx=200, Ny=200):
    xs = np.linspace(self.Xdomain[0], self.Xdomain[1], Nx)
    ys = np.linspace(self.Ydomain[0], self.Ydomain[1], Ny)
    input_data, target_data = np.meshgrid(xs, ys)
    return [input_data, target_data]

  def plot_sample_batch(self, batch_size=500):
      ids = np.random.choice(len(self), batch_size, replace=False)
      x_data = self.input_data[0][ids,:]
      y_data = self.input_data[1][ids,:]
      plt.scatter(x_data, y_data)
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('Sample batch = {}'.format(batch_size))
      plt.show()

  def plot_data(self):
      fig = plt.figure()
      for t, (t_idx, t_val) in zip(self.targets, self.target_data):
        x_data = self.input_data[0][t_idx,:]
        y_data = self.input_data[1][t_idx,:]
        plt.scatter(x_data, y_data, label=t, c=next(cycol))
      plt.xlabel('x')
      plt.ylabel('y')
      plt.legend(title="Training Data", bbox_to_anchor=(1.05, 1), loc='upper left')
      fig.tight_layout()
      plt.show()


class DataGeneratorXT:
  """ Generates 1D time-dependent collocation grid for training PINNs
  # Arguments:
    X: [X0, X1]
    T: [T0, T1]
    targets: list and type of targets you wish to impose on PINNs. 
        ('domain', 'ic', 'bc-left', 'bc-right', 'all')
    num_sample: total number of collocation points. 
    logT: generate random samples logarithmic in time. 

  # Examples: 
    >> dg = DataGeneratorXT([0., 1.], [0., 1.], ["domain", "ic", "bc-left", "bc-right"], 10000)
    >> input_data, target_data = dg.get_data()

  """
  def __init__(self, 
               X=[0., 1.],
               T=[0., 1.],
               targets=['domain', 'ic', 'bc-left', 'bc-right'], 
               num_sample=10000,
               logT=False):
    'Initialization'
    self.Xdomain = X
    self.Tdomain = T
    self.logT = logT
    self.targets = targets
    self.num_sample = num_sample
    self.input_data = None
    self.target_data = None
    self.set_data()

  def __len__(self):
    return self.input_data[0].shape[0]

  def set_data(self):
    self.input_data, self.target_data = self.generate_data()

  def get_data(self):
    return self.input_data, self.target_data

  def generate_uniform_T_samples(self, num_sample):
    if self.logT is True:
      t_dom = np.random.uniform(np.log1p(self.Tdomain[0]), np.log1p(self.Tdomain[1]), num_sample)
      t_dom = np.exp(t_dom) - 1.
    else:
      t_dom = np.random.uniform(self.Tdomain[0], self.Tdomain[1], num_sample)
    return t_dom

  def generate_data(self):
    # Half of the samples inside the domain.
    num_sample = int(self.num_sample/2)
    
    counter = 0
    # domain points 
    x_dom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
    t_dom = self.generate_uniform_T_samples(num_sample)
    ids_dom = np.arange(x_dom.shape[0])
    counter += ids_dom.size

    # The other half distributed equally between BC and IC.
    num_sample = int(self.num_sample/4)

    # initial conditions
    x_ic = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
    t_ic = np.full(num_sample, self.Tdomain[0])
    ids_ic = np.arange(x_ic.shape[0]) + counter 
    counter += ids_ic.size

    # bc points 
    num_sample_per_edge = int(num_sample/2)
    # left bc points 
    x_bc_left = np.full(num_sample_per_edge, self.Xdomain[0])
    t_bc_left = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_left = np.arange(x_bc_left.shape[0]) + counter
    counter += ids_bc_left.size

    # right bc points 
    x_bc_right = np.full(num_sample-num_sample_per_edge, self.Xdomain[1])
    t_bc_right = self.generate_uniform_T_samples(num_sample-num_sample_per_edge)
    ids_bc_right = np.arange(x_bc_right.shape[0]) + counter 
    counter += ids_bc_right.size

    ids_bc = np.concatenate([ids_bc_left, ids_bc_right])
    ids_all = np.concatenate([ids_dom, ids_ic, ids_bc])

    ids = {
        'domain': ids_dom, 
        'bc-left': ids_bc_left, 
        'bc-right': ids_bc_right,
        'ic': ids_ic,
        'bc': ids_bc, 
        'all': ids_all
    }

    assert all([t in ids.keys() for t in self.targets]), \
      'accepted target types: {}'.format(ids.keys())

    input_data = [
        np.concatenate([x_dom, x_ic, x_bc_left, x_bc_right]).reshape(-1,1),
        np.concatenate([t_dom, t_ic, t_bc_left, t_bc_right]).reshape(-1,1),
    ]
    total_sample = input_data[0].shape[0]

    target_data = []
    for i, tp in enumerate(self.targets):
      target_data.append(
          (ids[tp], 'zeros')
      )
      
    return input_data, target_data

  def get_test_grid(self, Nx=200, Nt=200):
    xs = np.linspace(self.Xdomain[0], self.Xdomain[1], Nx)
    if self.logT:
      ts = np.linspace(np.log1p(self.Tdomain[0]), np.log1p(self.Tdomain[1]), Nt)
      ts = np.exp(ts) - 1.0
    else:
      ts = np.linspace(self.Tdomain[0], self.Tdomain[1], Nt)
    return np.meshgrid(xs, ts)

  def plot_sample_batch(self, batch_size=500):
      ids = np.random.choice(len(self), batch_size, replace=False)
      x_data = self.input_data[0][ids,:]
      t_data = self.input_data[1][ids,:]
      plt.scatter(x_data, t_data)
      plt.xlabel('x')
      plt.ylabel('t')
      plt.title('Sample batch = {}'.format(batch_size))
      plt.show()

  def plot_data(self):
      fig = plt.figure()
      for t, (t_idx, t_val) in zip(self.targets, self.target_data):
        x_data = self.input_data[0][t_idx,:]
        t_data = self.input_data[1][t_idx,:]
        plt.scatter(x_data, t_data, label=t, c=next(cycol))
      plt.xlabel('x')
      plt.ylabel('t')
      plt.legend(title="Training Data", bbox_to_anchor=(1.05, 1), loc='upper left')
      fig.tight_layout()
      plt.show()


class DataGeneratorXYT:
  """ Generates 2D time-dependent collocation grid for training PINNs
  # Arguments:
    X: [X0, X1]
    Y: [Y0, Y1]
    T: [T0, T1]
    targets: list and type of targets you wish to impose on PINNs. 
        ('domain', 'ic', 'bc-left', 'bc-right', 'bc-bot', 'bc-top', 'all')
    num_sample: total number of collocation points. 
    logT: generate random samples logarithmic in time. 

  # Examples: 
    >> dg = DataGeneratorXYT([0., 1.], [0., 1.], [0., 1.], 
                             ["domain", "ic", "bc-left", "bc-right", "bc-bot", "bc-top"], 
                             10000)
    >> input_data, target_data = dg.get_data()

  """
  def __init__(self, 
               X=[0., 1.],
               Y=[0., 1.],
               T=[0., 1.],
               targets=['domain', 'ic', 'bc-left', 'bc-right', 'bc-bot', 'bc-top'], 
               num_sample=10000,
               logT=False):
    'Initialization'
    self.Xdomain = X
    self.Ydomain = Y
    self.Tdomain = T
    self.logT = logT
    self.targets = targets
    self.num_sample = num_sample
    self.input_data = None
    self.target_data = None
    self.set_data()

  def __len__(self):
    return self.input_data[0].shape[0]

  def set_data(self):
    self.input_data, self.target_data = self.generate_data()

  def get_data(self):
    return self.input_data, self.target_data

  def generate_uniform_T_samples(self, num_sample):
    if self.logT is True:
      t_dom = np.random.uniform(np.log1p(self.Tdomain[0]), np.log1p(self.Tdomain[1]), num_sample)
      t_dom = np.exp(t_dom) - 1.
    else:
      t_dom = np.random.uniform(self.Tdomain[0], self.Tdomain[1], num_sample)
    return t_dom

  def generate_data(self):
    # Half of the samples inside the domain.
    num_sample = int(self.num_sample/2)
    
    counter = 0
    # domain points 
    x_dom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
    y_dom = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample)
    t_dom = self.generate_uniform_T_samples(num_sample)
    ids_dom = np.arange(x_dom.shape[0])
    counter += ids_dom.size

    # The other half distributed equally between BC and IC.
    num_sample = int(self.num_sample/4)

    # initial conditions
    x_ic = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
    y_ic = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample)
    t_ic = np.full(num_sample, self.Tdomain[0])
    ids_ic = np.arange(x_ic.shape[0]) + counter 
    counter += ids_ic.size

    # bc points 
    num_sample_per_edge = int(num_sample/4)
    # left bc points 
    x_bc_left = np.full(num_sample_per_edge, self.Xdomain[0])
    y_bc_left = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample_per_edge)
    t_bc_left = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_left = np.arange(x_bc_left.shape[0]) + counter
    counter += ids_bc_left.size

    # right bc points 
    x_bc_right = np.full(num_sample_per_edge, self.Xdomain[1])
    y_bc_right = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample_per_edge)
    t_bc_right = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_right = np.arange(x_bc_right.shape[0]) + counter 
    counter += ids_bc_right.size

    # bot bc points 
    x_bc_bot = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample_per_edge)
    y_bc_bot = np.full(num_sample_per_edge, self.Ydomain[0])
    t_bc_bot = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_bot = np.arange(x_bc_bot.shape[0]) + counter
    counter += ids_bc_bot.size

    # right bc points 
    x_bc_top = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample-num_sample_per_edge)
    y_bc_top = np.full(num_sample-num_sample_per_edge, self.Ydomain[1])
    t_bc_top = self.generate_uniform_T_samples(num_sample-num_sample_per_edge)
    ids_bc_top = np.arange(x_bc_top.shape[0]) + counter
    counter += ids_bc_top.size

    ids_bc = np.concatenate([ids_bc_left, ids_bc_right, ids_bc_bot, ids_bc_top])
    ids_all = np.concatenate([ids_dom, ids_ic, ids_bc])

    ids = {
        'domain': ids_dom, 
        'bc-left': ids_bc_left, 
        'bc-right': ids_bc_right,
        'bc-bot': ids_bc_bot,
        'bc-top': ids_bc_top,
        'ic': ids_ic,
        'bc': ids_bc, 
        'all': ids_all
    }

    assert all([t in ids.keys() for t in self.targets]), \
      'accepted target types: {}'.format(ids.keys())

    input_grid = [
        np.concatenate([x_dom, x_ic, x_bc_left, x_bc_right, x_bc_bot, x_bc_top]).reshape(-1,1),
        np.concatenate([y_dom, y_ic, y_bc_left, y_bc_right, y_bc_bot, y_bc_top]).reshape(-1,1),
        np.concatenate([t_dom, t_ic, t_bc_left, t_bc_right, t_bc_bot, t_bc_top]).reshape(-1,1),
    ]
    total_sample = input_grid[0].shape[0]

    target_grid = []
    for i, tp in enumerate(self.targets):
      target_grid.append(
          (ids[tp], 'zeros')
      )
      
    return input_grid, target_grid

  def get_test_grid(self, Nx=50, Ny=50, Nt=100):
    xs = np.linspace(self.Xdomain[0], self.Xdomain[1], Nx)
    ys = np.linspace(self.Ydomain[0], self.Ydomain[1], Ny)
    if self.logT:
      ts = np.linspace(np.log1p(self.Tdomain[0]), np.log1p(self.Tdomain[1]), Nt)
      ts = np.exp(ts) - 1.0
    else:
      ts = np.linspace(self.Tdomain[0], self.Tdomain[1], Nt)
    return np.meshgrid(xs, ys, ts)

  def plot_sample_batch(self, batch_size=500):
      ids = np.random.choice(len(self), batch_size, replace=False)
      x_data = self.input_data[0][ids,:]
      y_data = self.input_data[1][ids,:]
      t_data = self.input_data[2][ids,:]
      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      ax.scatter(x_data, y_data, t_data)
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('t')
      plt.title('Sample batch = {}'.format(batch_size))
      plt.show()

  def plot_data(self):
      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      for t, (t_idx, t_val) in zip(self.targets, self.target_data):
        x_data = self.input_data[0][t_idx,:]
        y_data = self.input_data[1][t_idx,:]
        t_data = self.input_data[2][t_idx,:]
        ax.scatter(x_data, y_data, t_data, label=t, c=next(cycol))
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('t')
      plt.legend(title="Training Data", bbox_to_anchor=(1.05, 1), loc='upper left')
      fig.tight_layout()
      plt.show()


def ex1():
    dg = DataGeneratorX(
        X=[-1., 1.],
        targets=['domain', 'bc-left', 'bc-right'], 
        num_sample=1000
    )
    dg.plot_data()
    dg.plot_sample_batch(100)

def ex2():
    dg = DataGeneratorXY(
        X=[-1., 1.],
        Y=[0., 10.],
        targets=['domain', 'bc-left', 'bc-right', 'bc-bot', 'bc-top'], 
        num_sample=1000
    )
    dg.plot_data()
    dg.plot_sample_batch(100)

def ex3():
    dg = DataGeneratorXT(
        X=[-1., 1.],
        T=[0., 100.],
        targets=['domain', 'ic', 'bc-left', 'bc-right'], 
        num_sample=1000,
        logT=False
    )
    dg.plot_data()
    dg.plot_sample_batch(100)

def ex4():
    dg = DataGeneratorXYT(
        X=[-1., 1.],
        Y=[-1., 1.],
        T=[0., 100.],
        targets=['domain', 'ic', 'bc-left', 'bc-right', 'bc-bot', 'bc-top'],
        num_sample=2000,
        logT=False
    )
    dg.plot_data()
    dg.plot_sample_batch(500)


if __name__=='__main__':
  # ex1()
  # ex2()
  ex3()
  # ex4()

