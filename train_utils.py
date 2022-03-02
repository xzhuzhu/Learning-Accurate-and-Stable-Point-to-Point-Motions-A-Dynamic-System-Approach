from __future__ import print_function
import torch
from torch.utils.data import DataLoader, Dataset
import time
import copy
import os
import torch.optim as optim



def train(model, loss_fn, opt, train_dataset,
		  n_epochs=500, batch_size=None, shuffle=True,
		  clip_gradient=True, clip_value_grad=0.1,
		  clip_weight=False, clip_value_weight=2,
		  log_freq=5, logger=None, loss_clip=1e3, stop_threshold=float('inf')):
	'''
	train the torch model with the given parameters
	:param model (torch.nn.Module): the model to be trained
	:param loss_fn (callable): loss = loss_fn(y_pred, y_target)
	:param opt (torch.optim): optimizer
	:param x_train (torch.Tensor): training data (position/position + velocity)
	:param y_train (torch.Tensor): training label (velocity/control)
	:param n_epochs (int): number of epochs
	:param batch_size (int): size of minibatch, if None, train in batch
	:param shuffle (bool): whether the dataset is reshuffled at every epoch
	:param clip_gradient (bool): whether the gradients are clipped
	:param clip_value_grad (float): the threshold for gradient clipping
	:param clip_weight (bool): whether the weights are clipped (not implemented)
	:param clip_value_weight (float): the threshold for weight clipping (not implemented)
	:param log_freq (int): the frequency for printing loss and saving results on tensorboard
	:param logger: the tensorboard logger
	:return: None
	'''

	# if batch_size is None, train in batch
	scheduler = optim.lr_scheduler.StepLR(opt,10, gamma=0.999)

	n_samples = len(train_dataset)
	if batch_size is None:
		train_loader = DataLoader(
			dataset=train_dataset,
			batch_size=n_samples,
			shuffle=shuffle
		)
	else:
		train_loader = DataLoader(
			dataset=train_dataset,
			batch_size=batch_size,
			shuffle=shuffle
		)

	# record time elasped
	ts = time.time()

	if loss_fn.reduction == 'mean':
		mean_flag = True
	else:
		mean_flag = False

	best_train_loss = float('inf')
	best_train_epoch = 0
	best_model = model

	# train the model
	model.train()
	for epoch in range(n_epochs):

		# iterate over minibatches
		train_loss = 0.
		for x_batch, y_batch in train_loader:
			x_batch = x_batch.unsqueeze(0)
			y_batch = y_batch
			# forward pass
			if isinstance(x_batch, torch.Tensor):
				ls = model(x_batch)
				y_pred = ls[0].to(x_batch.device)
				weight = ls[1].to(x_batch.device)
				y_pred2 = ls[2].to(x_batch.device)
			elif isinstance(x_batch, dict):
				y_pred = model(**x_batch).to(x_batch.device)
			else:
				raise ValueError
			# compute loss
			loss =loss_fn(y_pred, y_batch.unsqueeze(0))+0.0001*loss_fn(y_pred2, torch.mm(y_batch,weight).unsqueeze(0)) #

			train_loss += loss.item()

			if loss > loss_clip:
				print('loss too large, skip')
				continue

			# backward pass
			opt.zero_grad()
			loss.backward()

			# clip gradient based on norm
			if clip_gradient:
				# torch.nn.utils.clip_grad_value_(
				#     model.parameters(),
				#     clip_value_grad
				# )
				torch.nn.utils.clip_grad_norm_(
					model.parameters(),
					clip_value_grad
				)
			# update parameters
			opt.step()

		if mean_flag:   # fix for taking mean over all data instead of mini batch!
			train_loss = float(batch_size)/float(n_samples)*train_loss

		if epoch - best_train_epoch >= stop_threshold:
			break

		if train_loss < best_train_loss:
			best_train_epoch = epoch
			best_train_loss = train_loss
			best_model = copy.deepcopy(model)

		# report loss in command line and tensorboard every log_freq epochs
		if epoch % log_freq == (log_freq - 1):
			print('    Epoch [{}/{}]: current loss is {}, time elapsed {} second'.format(
				epoch + 1, n_epochs,
				train_loss,
				time.time()-ts)
			)

			if logger is not None:
				info = {'Training Loss': train_loss}

				# log scalar values (scalar summary)
				for tag, value in info.items():
					logger.scalar_summary(tag, value, epoch + 1)

				# log values and gradients of the parameters (histogram summary)
				for tag, value in model.named_parameters():
					tag = tag.replace('.', '/')
					logger.histo_summary(
						tag,
						value.data.cpu().numpy(),
						epoch + 1
					)
					logger.histo_summary(
						tag + '/grad',
						value.grad.data.cpu().numpy(),
						epoch + 1
					)
		scheduler.step()

	return best_model, best_train_loss


def test(model, loss_fn, x_test, y_test):
	'''
	test the torch model
	:param: model (torch.nn.Model): the trained torch model
	:param: loss_fn (callable): loss=loss_fn(y_pred, y_target)
	:param: x_test (torch.Tensor): test input
	:param: y_test (torch.Tensor): test target output
	:return: loss (float): loss over the test set
	'''
	model.eval()
	y_pred = model(x_test)

	return loss_fn(y_pred, y_test).item()

