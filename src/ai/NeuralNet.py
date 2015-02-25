"""

NeuralNet class creates a neural network.

"""
import os
import numpy as np
import time
from collections import OrderedDict
from caffe import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class NeuralNet(SGDSolver):
	

	#dummy_filters = np.ones((32,18,1,1), dtype=np.float32)
	dummy_filters = None

	def __init__(self, solver_file, model_file, pretrained_file=None, gpu=False):
		"""
		Initialize a NeuralNet
		@param solver_file: path to solver prototxt file
		@param pretrained_file: path to pretrained model (optional)
		@param model_file: path to net model
		@param gpu: run on GPU?
		"""
				
		SGDSolver.__init__(self,solver_file)

		if pretrained_file != None:	
			self.net.copy_from(pretrained_file)

		if gpu:
			set_mode_gpu()
		else:
			set_mode_cpu()

		df = np.tile(np.array([1000,1000,0,1000,1000,0,0,0,0,0,0,1000,1000,0,0,0,0,0], dtype=np.float32), (32,1))
		self.dummy_filters = df.reshape((32,18,1,1))

		#print self.dummy_filters[0]

	def train(self, inputs, outputs):
		"""
		Use the SGDSolver.Solve to optimize the network

		@param inputs: NxM numpy.ndarray, where N is number of inputs and M is batch size
		@param outputs: KxM numpy.ndarray, where K is number of outputs and M is batch size
		@return cost?
		"""

		#assert inputs.shape[0] == self.nr_inputs
		#assert outputs.shape[0] == self.nr_outputs
		#assert inputs.shape[1] == outputs.shape[1]
		#inputs = image data, outputs = qvalues?

		labels = outputs.reshape((32,18,1,1))

		dummy_labels = np.zeros((32,1,1,1), dtype=np.float32)

		dummy_filters = np.ones((32,18,1,1), dtype=np.float32)
		d = np.transpose(inputs)
		data = np.ascontiguousarray(d.reshape((32,4,84,84), order='C'), dtype=np.float32)

		self.net.set_input_arrays_(0, data, dummy_labels)		
		self.net.set_input_arrays_(1, labels.astype(np.float32), dummy_labels)
		self.net.set_input_arrays_(2, self.dummy_filters, dummy_labels)
		
		cost = self.step(1)

		return cost

	def predict(self, inputs):
		"""
		Predict neural network output layer activations for input.
		@param inputs: NxM numpy.ndarray, where N is number of inputs and M is batch size
		"""
		#we're given an 22842 x 32 array of inputs
		#for each column, create a contiguous array of 4x1x84x84
		dummy_data = np.zeros((32,18,1,1), dtype=np.float32)
		dummy_labels = np.zeros((32,1,1,1), dtype=np.float32)
		dummy_filters = np.ones((32,18,1,1), dtype=np.float32)
		#print np.shape(inputs)

		if np.shape(inputs)[1] == 1:
			d = np.transpose(inputs)
			da = np.ascontiguousarray(d.reshape((1,4,84,84), order='C'), dtype=np.float32)
			data = np.tile(da, (32,1,1,1))

			self.net.set_input_arrays_(0, data, dummy_labels)
			self.net.set_input_arrays_(1, dummy_data, dummy_labels)
			self.net.set_input_arrays_(2, self.dummy_filters, dummy_labels)
			outputs = self.net.forward()
			out = self.net.blobs['filtered_q_values'].data[0,:]
		else:		
			d = np.transpose(inputs)
			data = np.ascontiguousarray(d.reshape((32,4,84,84), order='C'), dtype=np.float32)
			self.net.set_input_arrays_(0, data, dummy_labels)
			self.net.set_input_arrays_(1, dummy_data, dummy_labels)
			self.net.set_input_arrays_(2, self.dummy_filters, dummy_labels)
			outputs = self.net.forward()
			out = self.net.blobs['filtered_q_values'].data		


		#print np.shape(data)
		#test = raw_input("Press Enter to continue...")
		#matplotlib to see if it makes sense -> imshow(test[0,2,:])
		#print np.shape(np.transpose(inputs))
		#print test[0,2,:]
		#plt.imshow(data[0,1,:], cmap = cm.Greys_r)
		#plt.savefig('tessstttyyy.png', dpi=72)


		return out

	def get_weight_stats(self):
		# copy weights from GPU to CPU memory
        
		return None

	def save_network(self, epoch):
		self.epoch = epoch
		self.net.save('ai/models/deepmind_caffe.caffemodel')
		return None
