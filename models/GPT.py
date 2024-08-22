from openai import AzureOpenAI
from .Model import Model

class GPT(Model):
    def __init__(self, model_config):
        super().__init__(model_config)
        api_keys = model_config["api_key_info"]["api_keys"]
        endpoint = model_config["api_key_info"]["azure_endpoint"]
        api_version = model_config["api_key_info"]["api_version"]
        self.max_tokens = int(model_config["params"]["max_output_tokens"])
        self.temperature = float(model_config["params"]["temperature"])
        self.client = AzureOpenAI(
            api_key=api_keys,
            azure_endpoint=endpoint,
            api_version=api_version
        )

    def get_completion(self, prompt):

        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response.choices[0].message.content
