

from typing import List, Union, Dict
from numpy import ndarray

def dict_to_list_hierarchy(params) -> List[List[ndarray]]:
    result = []
    for key, value in params.items():
        if isinstance(value, dict):
            result.extend(dict_to_list_hierarchy(value))
        else:
            result.append(value)
    return result


def merge_args(x: Union[ndarray, tuple], model_parameters: Union[List, Dict]) -> List:
    new_inputs = list()
    if isinstance(model_parameters, dict):
        model_parameters = dict_to_list_hierarchy(model_parameters)
    new_inputs.append(x)
    new_inputs.extend(model_parameters)
    return new_inputs


def verify_output(actual_output, estimated_output):
    #todo
    pass

def verify_adjoints(actual_adjoint, estimated_adjiont):
    #todo
    multiple_active_para =  isinstance(estimated_adjiont, tuple)
    if not multiple_active_para:
        pass
    else:
        pass

    pass

