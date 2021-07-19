import functools
import collections

import torch
import torch.nn as nn


def long_sum(v):
    if not all(map(lambda x: isinstance(x, int), v)):
        raise ValueError('The long_sum only supports the sequence with all int elements.')
    return functools.reduce(lambda x, y: x + y, v)


def long_prod(v):
    if not all(map(lambda x: isinstance(x, int), v)):
        raise ValueError('The long_prod only supports the sequence with all int elements.')
    return functools.reduce(lambda x, y: x * y, v)


def get_recursive_total_size(object_with_size):
    if isinstance(object_with_size, dict):
        return long_sum(map(get_recursive_total_size, object_with_size.values()))
    elif isinstance(object_with_size, int):
        return object_with_size
    elif isinstance(object_with_size, list):
        return long_sum(list(map(get_recursive_total_size, object_with_size)))
    elif isinstance(object_with_size, tuple) and not isinstance(object_with_size[0], int):
        return long_sum(list(map(get_recursive_total_size, object_with_size)))
    else:
        return long_prod(object_with_size)


def get_recursive_shape(object_with_shape):
    if isinstance(object_with_shape, dict):
        return {key: get_recursive_shape(sub_object) for key, sub_object in object_with_shape.items()}
    elif isinstance(object_with_shape, list):
        return list(map(get_recursive_shape, object_with_shape))
    elif isinstance(object_with_shape, tuple) and not isinstance(object_with_shape[0], int):
        return tuple(map(get_recursive_shape, object_with_shape))
    else:
        return object_with_shape.size()


def get_layer_formatted_summary(layer_name, layer):
    return get_layer_formatted_summary_explicit(layer_name, layer['output_shape'], layer['nb_params'])


def get_layer_formatted_summary_explicit(layer_name, output_shape, nb_params):
    details = get_recursive_layer_details(layer_name, output_shape, '{0:,}'.format(nb_params))
    return get_details_formatted_summary(details)


def get_details_formatted_summary(details):
    return '\n'.join([format_layer_summary(*layer_details) for layer_details in details])


def get_recursive_layer_details(layer_name, output_shape, nb_params):
    if isinstance(output_shape, dict):
        sub_lines = [(get_recursive_layer_details(key+': ', sub_output, ''))
                     for key, sub_output in output_shape.items()]
        sub_lines = [line for lines in sub_lines for line in lines]
        return [get_layer_details(layer_name, 'dict', nb_params),
                get_layer_details('', '{', ''),
                *sub_lines,
                get_layer_details('', '}', '')]
    elif isinstance(output_shape, list):
        sub_lines = [get_recursive_layer_details('', sub_output, '') for sub_output in output_shape]
        sub_lines = [line for lines in sub_lines for line in lines]
        return [get_layer_details(layer_name, 'list', nb_params),
                get_layer_details('', '[', ''),
                *sub_lines,
                get_layer_details('', ']', '')]
    elif isinstance(output_shape, tuple) and not isinstance(output_shape[0], int):
        sub_lines = [get_recursive_layer_details('', sub_output, '') for sub_output in output_shape]
        sub_lines = [line for lines in sub_lines for line in lines]
        return [get_layer_details(layer_name, 'tuple', nb_params),
                get_layer_details('', '(', ''),
                *sub_lines,
                get_layer_details('', ')', '')]
    else:
        return [get_layer_details(layer_name, output_shape, nb_params)]


def get_layer_details(layer_name, output_shape, nb_params):
    return format_layer_name(layer_name), \
           str(output_shape), \
           nb_params


def format_layer_name(layer_name):
    if len(layer_name) > 20:
        layer_name = '{lhead}...{ltail}'.format(lhead=layer_name[:8], ltail=layer_name[-9:])  # 20 = 9 + 8 + 3
    return layer_name


def format_layer_summary(layer_display, output_shape, nb_params):
    line_new = "{:>20}      {:<25} {:>15}".format(
        layer_display,
        str(output_shape),
        nb_params,
    )
    return line_new


def summary(model, input_size, batch_size=-1, device='cuda:0', dtypes=None):
    '''Keras-style torch summary
    Iterate the whole pytorch model and summarize the infomation as a Keras-style
    text report. The output would be store in a str.
    Arguments:
        model: an instance of nn.Module
        input_size: a sequence (list/tuple) or a sequence of sequnces, indicating
                    the size of the each model input variable.
        batch_size: a int. The batch size used for testing and displaying the
                    results.
        device: a str or torch.device. Should be set according to the deployed
                device of the argument "model".
        dtype: a list or torch data type for each input variable.
    Returns:
        1. tensor, total parameter numbers.
        2. tensor, trainable parameter numbers.
    '''
    if isinstance(device, str):
        device = torch.device(device)
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device='cuda:0', dtypes=None):
    '''Keras-style torch summary (string output)
    Iterate the whole pytorch model and summarize the infomation as a Keras-style
    text report. The output would be store in a str.
    Arguments:
        model: an instance of nn.Module
        input_size: a sequence (list/tuple) or a sequence of sequnces, indicating
                    the size of the each model input variable.
        batch_size: a int. The batch size used for testing and displaying the
                    results.
        device: a str or torch.device. Should be set according to the deployed
                device of the argument "model".
        dtype: a list or torch data type for each input variable.
    Returns:
        1. str, the summary text report.
        2. tuple, (total parameter numbers, trainable parameter numbers)
    '''
    if isinstance(device, str):
        device = torch.device(device)

    summary_str = ''

    def register_hook(module):
        def hook(hooked_module, module_input, output):
            class_name = str(hooked_module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(full_summary)

            m_key = '{name:s}-{idx:d}'.format(name=class_name, idx=module_idx + 1)
            layer_summary = collections.OrderedDict()
            layer_summary["input_shape"] = get_recursive_shape(module_input)
            layer_summary["output_shape"] = get_recursive_shape(output)

            params = 0
            params_trainable = 0
            for param in hooked_module.parameters(recurse=False):
                nb_param = torch.prod(torch.LongTensor(list(param.size()))).item()
                params += nb_param
                params_trainable += nb_param if param.requires_grad else 0
            layer_summary["nb_params"] = params
            layer_summary["nb_params_trainable"] = params_trainable
            full_summary[m_key] = layer_summary

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, (list, tuple)) and len(input_size) > 0:
        if not isinstance(input_size[0], (list, tuple)):
            input_size = (input_size, )
    else:
        raise ValueError('The argument "input_size" is not a tuple of a sequence of tuple. Given "{0}".'
                         .format(input_size))

    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)
    if len(dtypes) != len(input_size):
        raise ValueError('The lengths of the arguments "input_size" and "dtypes" does not correspond to each other.')

    # batch_size of 2 for batchnorm
    if batch_size == -1:
        batch_size_ = 2
    else:
        batch_size_ = batch_size
    x = tuple(torch.rand(batch_size_, *in_size).type(dtype).to(device=device)
              for in_size, dtype in zip(input_size, dtypes))

    # create properties
    full_summary = collections.OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "--------------------------------------------------------------------" + "\n"
    line_new = format_layer_summary("Layer (type)", "Result Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "====================================================================" + "\n"
    for i, current_input in enumerate(x):
        summary_str += get_layer_formatted_summary_explicit('Input-' + chr(65+i),
                                                            get_recursive_shape(current_input),
                                                            0) \
                       + '\n'
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer_name in full_summary:
        # input_shape, output_shape, trainable, nb_params
        sum_layer = full_summary[layer_name]
        line_new = get_layer_formatted_summary(layer_name, sum_layer)
        total_params += sum_layer["nb_params"]

        output_shape = sum_layer["output_shape"]
        total_output += get_recursive_total_size(output_shape)
        trainable_params += sum_layer["nb_params_trainable"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(long_sum(list(map(long_prod, input_size))) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "====================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    summary_str += "--------------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "--------------------------------------------------------------------"
    # return summary
    return summary_str, (total_params, trainable_params)
