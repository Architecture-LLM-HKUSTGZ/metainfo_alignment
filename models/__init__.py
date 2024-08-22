import json
from .GPT import GPT

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def init_model_config(model_config_path):
    # 大模型参数配置初始化
    model_config = load_json(model_config_path)
    api_provider = model_config['model_info']['provider']

    if api_provider == 'azure_openai':
        model = GPT(model_config)

    else:
        raise ValueError(f"Error: unknown api_provider {api_provider}.")

    return model
