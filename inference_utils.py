import torch
from time import time

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

def test_accuracy(model, testLoader):
    model.eval()
    running_acc = 0
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(testLoader):
            samples, labels = data

            outputs = model(samples)
            preds = torch.argmax(outputs, 1)

            running_acc += torch.sum(preds == labels)
            num_samples += samples.size(0)
    
    return running_acc / num_samples

def test_speed(model):

    dummy_sample = torch.randn((1,3,224,224))

    # Average out inference speed over multiple iterations 
    # to get a true estimate
    num_iterations = 100
    start = time()
    for _ in range(num_iterations):
        _ = model(dummy_sample)

    end = time()
    return (end-start)/num_iterations * 1000

def load_quantized_model(model_to_quantize, weights_path):

    '''
    Model only needs to be calibrated for the first time.
    Next time onwards, to load the quantized model, 
    you still need to prepare and convert the model without calibrating it.
    After that, load the state dict as usual.
    '''
    model_to_quantize.eval()

    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    model_prep = prepare_fx(model_to_quantize, qconfig_mapping, 
                            torch.randn((1,3,224,224)))
    quantized_model = convert_fx(model_prep)
    quantized_model.load_state_dict(torch.load(weights_path))

    return quantized_model