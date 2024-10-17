import torch
import numpy
import torchvision
#from torchinfo import summary
from torchsummary import summary
#from torchsummary import summary_string

f = open('Linear.txt','w')

def linear_model(dataset, num_classes=10):
    """Define the simplest linear model."""
    if 'cifar' in dataset.lower():
        dimension = 150528
    elif 'mnist' in dataset.lower():
        dimension = 784
    elif 'imagenet' in dataset.lower():
        dimension = 150528
    elif 'tinyimagenet' in dataset.lower():
        dimension = 64**2 * 3
    else:
        raise ValueError('Linear model not defined for dataset.')
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(dimension, num_classes))

def model_param_extract(model_name):
	model_name.cuda()
	model_summary = summary(model_name, [(3,224,224)])
	print(str(model_summary))
	#s = summary(model_name, (3,224,224))
	print(model_name, file=f)
	#file.write(str(s))
	#file.write(str(print(model_name)))

if __name__ == '__main__':

	model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
	model2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
	model3 = torchvision.models.vgg16()
	model4 = torchvision.models.convnext_base()
	model5 = linear_model('cifar',num_classes=100)

	#model_param_extract(model)
	#model_param_extract(model2)
	#model_param_extract(model3)
	#model_param_extract(model4)
	model_param_extract(model5)
