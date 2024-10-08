import torch
import numpy
import torchvision


def model_param_extract(model_name):
	count = 1
	param_sum = 0
	with open('Model_Parameters_Summary.txt','a') as f:
		f.write(f'\n\nparameters model ResNet 50 has\n\n\n\n\n')
		for parameter in model_name.parameters():
			f.write(f'layer {count} ' + str(len(parameter)) + '\n\n')
			param_sum += len(parameter)
			count+=1
		f.write(f'\n\n Total Parameters = {param_sum}')
		f.close()

if __name__ == '__main__':

	model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
	model2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
	model3 = torchvision.models.vgg16()

	#model_param_extract(model)
	#model_param_extract(model2)
	model_param_extract(model3)
