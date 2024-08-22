class Model:
    def __init__(self, model_config):
        self.api_provider = model_config["model_info"]["provider"]
        self.name = model_config["model_info"]["name"]
        self.temperature = float(model_config["params"]["temperature"])
        # self.gpus = [str(gpu) for gpu in model_config["params"]["gpus"]]

    def api_key_error(self):
        print(f"Api key provided by {self.api_provider} does not match the selected model {self.name}!")
