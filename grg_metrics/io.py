import json

def parse_grg_case_file(grg_file_name):
    '''opens the given path and parses it as json data

    Args:
        grg_file_name(str): path to the a json data file
    Returns:
        Dict: a dictionary case
    '''
    with open(grg_file_name, 'r') as grg_data:
        data = json.load(grg_data)
        grg_data.close()

    return data
