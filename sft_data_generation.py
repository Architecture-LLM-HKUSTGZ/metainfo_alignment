import os
import embedding
import faiss
from config import get_args
from tqdm import tqdm
from models import *


# plan stage
scenarios = 'xxx'
specification = 'xxx'

# write stage
how_to_write_a_method_statement = 'xxx'
generated_sections = 'xxx'

# long document generation strategy
plan_prompt = f'''
You are tasked with generating a detailed Method Statement (MS) for a specific construction scenario. Each construction scenario may involve different types of work, such as Civil Engineering Works, Trackwork, etc., and therefore, each requires a tailored MS structure.

To assist in generating the MS, I will provide the specific construction scenario and its corresponding construction specification document. Use this information, along with the "how to write a method statement" guide, to determine the appropriate sections for the MS.

The construction scenario is as follows:

{scenarios}

The corresponding construction specification document is as follows:

{specification}

Based on this scenario and specification document, please break down the Method Statement into multiple sections. Each section should be clearly defined and should include the main objectives, specific tasks, safety considerations, and any other relevant details. Also, include the expected word count for each section.

Use the following structure for the Method Statement, based on the key elements provided in the "how to write a method statement" guide:

1. **Introduction**: A brief overview of the task or activity that the method statement relates to.
2. **Scope of Work**: A clear definition of the work that will be carried out, including any constraints or limitations.
3. **Equipment and Materials**: A list of the equipment and materials that will be required to complete the task, including any specific safety requirements.
4. **Personnel**: A description of the personnel required to complete the task, including their roles and responsibilities.
5. **Sequence of Events**: A step-by-step guide to the sequence of events that will take place, including any critical path activities.
6. **Health and Safety Considerations**: A detailed overview of the health and safety considerations that must be taken into account when carrying out the task, including any relevant regulations or standards.
7. **Quality Assurance**: Information on how quality will be ensured, including any inspections or testing that will take place.
8. **Emergency Procedures**: Information on the procedures that should be followed in the event of an emergency.
9. **Conclusion**: A summary of the key points covered in the method statement, including any recommendations for future improvements.

Please ensure that each section is comprehensive and specific, with a word count that ensures clarity and detail without being excessive. The overall structure should follow the guidelines in the "how to write a method statement" guide.

'''


write_prompt = f'''
You are a skilled construction project writing assistant. Your task is to help generate a detailed Method Statement (MS) document for a specific construction scenario. I will provide you with the original writing instruction, the planned sections (writing steps), and the text that has already been written. Your job is to continue writing the next section based on this information.

Writing instruction:

{how_to_write_a_method_statement}

Planned sections:

{plan}

Already written text:

{generated_sections}

Now, please continue writing the next section, {STEP}, of the Method Statement. Ensure that the section aligns with the planned structure and content requirements, while also following the original writing instruction. You may add a relevant subtitle at the beginning of the section if necessary. Ensure that the content is clear, professional, and appropriate for a Method Statement. Remember to only output the new section you write, without repeating the already written text.

'''


def spec_summary(llm, relevant_source, k):
    '''根据施工规范生成对应的施工计划'''
    prompt = f'''
    
    '''

    responses = llm.get_completion(prompt)

    messages = [
        {"role": "usr", "content": plan_prompt},
        {"role": "system", "content": how_to_write_a_method_statement},
        {"role": "assistant", "content": "..."},
    ]

    messages
    return responses


def ms_plan():



def ms_write():



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
