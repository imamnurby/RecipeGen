import pandas as pd
import string


def get_function_mapping(df_metadata, split_ref):
    df_cp = df_metadata[['function', 'desc', 'split']].copy()
    df_cp.drop_duplicates(inplace=True)
    return {function:desc for function, desc, split in df_cp.values if split==split_ref}

def get_channel_mapping(df_metadata):
    df_cp = df_metadata[['channel', 'channel_desc']].copy()
    df_cp.drop_duplicates(inplace=True)
    return {channel:desc for channel, desc in df_cp.values}

def get_valid_field(df_metadata):
    df_cp = df_metadata[["function", "field"]].copy()
    df_cp.drop_duplicates(inplace=True)
    field_list = df_cp.field.tolist()
    function_list = df_cp.function.tolist()
    field_list_out=[]
    field_mapping={}
    for temp_field, temp_function in zip(field_list, function_list):
        temp_field = temp_field.split(" ### ")
        temp_field = [x.strip().lower() for x in temp_field if x.strip() != '']
        temp_field = [x.translate(str.maketrans(' ', '_', string.punctuation)) for x in temp_field]
        for item in temp_field:
            if item not in field_list_out:
                field_list_out.append(item)
        temp_field = "(" + ", ".join(temp_field) + ")"
        field_mapping[temp_function] = temp_field
    return field_list_out, field_mapping

def get_channel_to_function_mapping(df_metadata):
    df_cp = df_metadata[["channel", "function"]].copy()
    df_cp.drop_duplicates(inplace=True)
    out_dict={}
    for temp_channel, temp_function in df_cp.values:
        if temp_channel not in out_dict.keys():
            out_dict[temp_channel] = []
        temp_function = temp_function.split(".")[-1]
        out_dict[temp_channel].append(temp_function)
    out_dict = {k:", ".join(v) for k,v in out_dict.items()}
    return out_dict

def get_metadata(path):
    df_metadata = pd.read_csv(path)
    df_metadata = df_metadata.fillna('')
    function_dict_trigger = get_function_mapping(df_metadata=df_metadata, split_ref='trigger')
    function_dict_action = get_function_mapping(df_metadata=df_metadata, split_ref='action')
    channel_dict = get_channel_mapping(df_metadata=df_metadata)
    valid_field,field_mapping = get_valid_field(df_metadata=df_metadata)
    channel_to_function_dict = get_channel_to_function_mapping(df_metadata=df_metadata)
    return channel_dict, function_dict_trigger, function_dict_action, field_mapping, valid_field, channel_to_function_dict

def append_prefix(desc, prefix):
    return prefix + desc

def append_suffix(desc, suffix):
    return desc + suffix

def process_field(raw_field):
    field = raw_field.split(" ### ")
    field = [x.strip().lower() for x in field if x.strip() != '']
    field = [x.translate(str.maketrans(' ', '_', string.punctuation)) for x in field]
    return field