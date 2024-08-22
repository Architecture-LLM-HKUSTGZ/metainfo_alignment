import os
import embedding
import faiss
from config import get_args
from tqdm import tqdm
from models import *


def question_generation(llm, root_node, node, k):
    '''根据当前节点生成问题'''
    prompt = f'''
    Task Description:
    - We have a Method Statement for construction workflows in the architecture domain. Each Method Statement has been converted into a hierarchical JSON structure, where each level has its corresponding nodes (titles from the Method Statement document) and content. Now, I need you to generate {k} targeted questions based on the content of the current node.
    
    Here is the information for the current root node and sub-node:
    - Root node: '{root_node}'
    - Current sub-node content: '{node}'
    
    Based on the above node content, generate {k} relevant questions as if you were a construction engineering personnel.
    
    Example input (i.e. current sub-node content):
    - "1. **Introduction**": "China State Construction Engineering (Hong Kong) Limited (CSHK) has been awarded to carry out construction works for Siu Ho Wan Depot Property Development Oyster Bay Station and Associated Works. The principle methods described in the following sections will be subject to review during construction and may be amended if so required.  Scope of Works: The method statement details the procedures to be undertaken for the Control Survey, Initial Land Record Survey, Record and Progress Survey, Setting-out and Survey Check, As-built Record Survey, Settlement Monitoring Survey etc.",
    
    Example output:
    - "What is the scope of works for the Siu Ho Wan Depot Property Development Oyster Bay Station project?"
    - "Tell me what is the purpose of Method Statement for General Site Survey Works?"
    - "Who is responsible for the construction works at the Siu Ho Wan Depot Property Development?"
    '''

    questions = llm.get_completion(prompt)

    return questions


def response_generation(llm, questions, node, relevant_source, k):
    '''根据当前问题生成响应'''
    prompt = f'''
    Task Description:
    Based on the following {k} questions, generate the corresponding {k} responses and construct {k} JSON objects based on these questions and responses. Each JSON object should contain the following two keys:
    - "user": This part mimics the user's question based on the current node content (i.e., {node}), the aim of this part is to ask possible questions given by the user related to the method statement.
    - "response": Combine the provided questions and relevant information to answer the question in the "user", please present it in a list format.
    
    Here are the provided questions:
    - '{questions}'
    Here is the relevant information for the questions, which can be referenced for the responses:
    - '{relevant_source}'
    
    Example output:
    - {{"user": "xxx", "response": ["xxx"]}}
      
    Notes:
    - The generated {k} JSON data should be in single-line format for easy processing and analysis.
    - The output is in direct JSON format without any explanatory code.
    - Do not generate comments like ``json``.
    '''

    responses = llm.get_completion(prompt)

    return responses


def get_relevant_source(questions, json_data, embedding_model, top_k):
    questions_embedding = embedding_model.encode([questions]).astype('float32')
    index = faiss.read_index(args.knowledge_index)
    _, similar_indices = index.search(questions_embedding, top_k)

    result = []
    keys_list = list(json_data.keys())
    for idx in similar_indices[0]:
        if 0 <= idx < len(keys_list):
            key = keys_list[idx]
            result.append({key: json_data[key]})

    result_merge = json.dumps(result, indent=4)

    return result_merge


def generate_from_ms(llm, json_data, embedding_model, parent_title=""):
    '''遍历节点，根据生成的问题生成响应'''
    for key, value in tqdm(json_data.items(), desc=f"Processing current root node"):
        root_node = f"{parent_title} {key}".strip()     # 获取当前根节点

        for node in tqdm(value, desc="Processing current sub-node"):
            node_content = {node: value[node]}
            questions = question_generation(llm, root_node, node_content, args.k)
            relevant_source = get_relevant_source(questions, json_data, embedding_model, top_k=args.top_k)
            responses = response_generation(llm, questions, node_content, relevant_source, args.k)

            with open(os.path.join(args.data_result, 'generated_from_ms.jsonl'), 'a', encoding='utf-8') as file:
                full_text = ''.join(responses)
                file.write(full_text + '\n')


if __name__ == "__main__":
    args = get_args()

    # load LLM configuration
    if args.model_configs == None:
        args.model_configs = f"model_configs/{args.generative_model}.json"

    if not os.path.exists(args.data_result):
        os.makedirs(args.data_result)

    llm = init_model_config(args.model_configs)

    # generate finetune data from method statement
    if args.from_ms == "True":
        json_data = embedding.load_json_data(args.knowledge_source)
        embedding_model = embedding.load_embedding_model(args.embedding_model)
        generate_from_ms(llm=llm, json_data=json_data, embedding_model=embedding_model, parent_title="")
