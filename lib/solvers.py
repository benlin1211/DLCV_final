import logging

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau, MultiStepLR


class LambdaStepLR(LambdaLR):

  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v


class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""

  def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)


class SquaredLR(LambdaStepLR):
  """ Used for SGD Lars"""

  def __init__(self, optimizer, max_iter, last_step=-1):
    super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**2, last_step)


class ExpLR(LambdaStepLR):

  def __init__(self, optimizer, step_size, gamma=0.9, last_step=-1):
    # (0.9 ** 21.854) = 0.1, (0.95 ** 44.8906) = 0.1
    # To get 0.1 every N using gamma 0.9, N * log(0.9)/log(0.1) = 0.04575749 N
    # To get 0.1 every N using gamma g, g ** N = 0.1 -> N * log(g) = log(0.1) -> g = np.exp(log(0.1) / N)
    super(ExpLR, self).__init__(optimizer, lambda s: gamma**(s / step_size), last_step)


def initialize_optimizer(model, config, lr=None):

    assert config.optimizer in ['SGD', 'Adagrad', 'Adam', 'RMSProp', 'Rprop', 'SGDLars']

    params = model.parameters()
    model_name = type(model).__name__

    if lr is not None:
        learning_rate = lr
    else:
        learning_rate = config.lr

    if config.optimizer == 'SGD':
        return SGD(
            params,
            lr=learning_rate,
            momentum=config.sgd_momentum,
            dampening=config.sgd_dampening,
            weight_decay=config.weight_decay)
    elif config.optimizer == 'Adam':
        return Adam(
            params,
            lr=learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay)
    else:
        logging.error('Optimizer type not supported')
        raise ValueError('Optimizer type not supported')


def initialize_scheduler(optimizer, config, last_step=-1):
    if config.scheduler == 'StepLR':
        return StepLR(optimizer, step_size=config.step_size, gamma=config.step_gamma, last_epoch=last_step)
    if config.scheduler == 'MultiStepLR':
        return MultiStepLR(optimizer, milestones=config.multi_step_milestones, gamma=config.step_gamma)
    elif config.scheduler == 'PolyLR':
        return PolyLR(optimizer, max_iter=config.max_epoch, power=config.poly_power, last_step=last_step)
    elif config.scheduler == 'SquaredLR':
        return SquaredLR(optimizer, max_iter=config.max_iter, last_step=last_step)
    elif config.scheduler == 'ExpLR':
        return ExpLR(
            optimizer, step_size=config.exp_step_size, gamma=config.exp_gamma, last_step=last_step)
    elif config.scheduler == 'ReduceLROnPlateau':

        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='max', verbose=True,
            factor=config.step_gamma, patience=config.reduce_patience,
            min_lr=config.scheduler_min_lr)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': config.scheadule_monitor
        }

        return scheduler
    else:
        logging.error('Scheduler not supported')
