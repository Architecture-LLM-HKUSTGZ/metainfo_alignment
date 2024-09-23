import os
import json
from models import *
from config import get_args
from tqdm import tqdm


def generate_sft_data_with_api(llm, input_dir, output_file):
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历文件夹中的所有文件，并显示进度条
        for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
            filepath = os.path.join(input_dir, filename)

            # 读取每个文档内容
            with open(filepath, 'r', encoding='utf-8') as doc_file:
                content = doc_file.read()

            # 生成文档标题（可根据文件名或文件内容生成）
            title = os.path.splitext(filename)[0]

            # 模拟用户提问的节点
            node = title
            prompt = f'''
                Task Description:
                You are tasked with simulating a user who is requesting the generation of a construction-related document, specifically a Method Statement (MS). The user's task is to formulate a clear and precise instruction for generating a document that most accurately reflects the content provided.
                Based on the following content, create a detailed user instruction that would most likely lead to the generation of this document. The instruction should be realistic, reflecting the type of requests typically made by industry professionals who need specific and actionable documents.

                Content:
                - '{content}'

                The instruction should be clear and specific, and it should guide the process of generating a Method Statement that includes all necessary procedures, safety measures, materials, equipment, and any other relevant details mentioned in the content.

                Example Instruction:
                "I need to generate a detailed Method Statement (MS) for the construction of a new railway bridge over a river. The construction scenario involves Civil Engineering Works, and the corresponding construction specification document includes material specifications, load-bearing requirements, and environmental impact considerations."

                Notes:
                - The output should be a single, coherent instruction that clearly communicates the user’s request.
                - Ensure the instruction fully encapsulates the details provided in the content, so the resulting document will be highly accurate and relevant.
                '''

            # 调用llm生成用户问题和回答
            responses = llm.get_completion(prompt)

            # 构建JSON对象
            conversation = {
                "messages": [
                    {"role": "user", "content": responses},
                    {"role": "assistant", "content": content}
                ]
            }

            # 将每个对话单独写入文件中
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    args = get_args()

    if args.model_configs is None:
        args.model_configs = f"model_configs/{args.generative_model}.json"

    llm = init_model_config(args.model_configs)

    input_dir = 'results/aligned_files'
    output_file = 'sft_training_data.jsonl'  # 注意这里使用JSONL格式
    generate_sft_data_with_api(llm, input_dir, output_file)
