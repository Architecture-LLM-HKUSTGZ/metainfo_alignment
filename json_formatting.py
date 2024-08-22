import os
import re
import json
from tqdm import tqdm

def parse_document_structure(text, root_name):
    lines = text.split("\n")
    document_structure = {}
    current_section = None
    current_subsection = None
    current_subsubsection = None
    current_section_content = ""
    current_subsection_content = ""
    current_subsubsection_content = ""

    for line in lines:
        line = line.strip()
        if re.match(r'^#\s', line):
            # New top-level title (e.g., # Engineering Document For Works)
            root_name = line[2:].strip()
            current_section = None
            current_subsection = None
            current_subsubsection = None
            current_section_content = ""
            document_structure = {}
        elif re.match(r'^##\s', line):
            # New section (e.g., ## Description or ## 1. Introduction)
            if current_subsection and current_subsubsection:
                document_structure[current_section][current_subsection][current_subsubsection] = current_subsubsection_content.strip()
            elif current_subsection:
                document_structure[current_section][current_subsection] = current_subsection_content.strip()
            elif current_section:
                document_structure[current_section] = current_section_content.strip()

            section_title = line[3:].strip()
            current_section = section_title
            document_structure[current_section] = {}
            current_subsection = None
            current_subsubsection = None
            current_section_content = ""
        elif re.match(r'^###\s', line):
            # New subsection (e.g., ### Description or ### 1.1. Subsection)
            if current_subsection and current_subsubsection:
                document_structure[current_section][current_subsection][current_subsubsection] = current_subsubsection_content.strip()
            elif current_subsection:
                document_structure[current_section][current_subsection] = current_subsection_content.strip()

            subsection_title = line[4:].strip()
            current_subsection = subsection_title
            document_structure[current_section][current_subsection] = {}
            current_subsubsection = None
            current_subsection_content = ""
        elif re.match(r'^####\s', line):
            # New subsubsection (e.g., #### Description or #### 1.1.1. Subsubsection)
            if current_subsubsection:
                document_structure[current_section][current_subsection][current_subsubsection] = current_subsubsection_content.strip()

            subsubsection_title = line[5:].strip()
            current_subsubsection = subsubsection_title
            document_structure[current_section][current_subsection][current_subsubsection] = {}
            current_subsubsection_content = ""
        elif current_subsubsection:
            # Append to current subsubsection
            current_subsubsection_content += line + " "
        elif current_subsection:
            # Append to current subsection
            current_subsection_content += line + " "
        elif current_section:
            # Append to current section
            current_section_content += line + " "

    # Assign the last accumulated content
    if current_section and current_subsection and current_subsubsection:
        document_structure[current_section][current_subsection][current_subsubsection] = current_subsubsection_content.strip()
    elif current_section and current_subsection:
        document_structure[current_section][current_subsection] = current_subsection_content.strip()
    elif current_section:
        document_structure[current_section] = current_section_content.strip()

    return {root_name: document_structure}

def format_to_markdown(structure, level=1):
    markdown_lines = []
    for key, value in structure.items():
        markdown_lines.append(f"{'#' * level} {key}")
        if isinstance(value, dict):
            markdown_lines.extend(format_to_markdown(value, level + 1))
        else:
            markdown_lines.append(value)
    return markdown_lines

def read_file(path):
    with open(path, "r", encoding='utf-8') as file:
        content = file.read()
    return content

def save_file(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding='utf-8') as file:
        file.write(content)

def save_json(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding='utf-8') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    llm_responses_dir = 'results/aligned_files'
    formatted_responses_dir = 'results/parsed_fine_grained_structure'
    markdown_responses_dir = 'results/markdown_files'

    # Ensure the output directories exist
    os.makedirs(formatted_responses_dir, exist_ok=True)
    os.makedirs(markdown_responses_dir, exist_ok=True)

    # Get a list of all .md files in the LLM responses directory
    llm_files = [f for f in os.listdir(llm_responses_dir) if f.endswith('.md')]

    # Process each file
    for llm_file in tqdm(llm_files, desc="Formatting LLM responses"):
        llm_file_path = os.path.join(llm_responses_dir, llm_file)
        llm_response = read_file(llm_file_path)

        root_name = llm_file.replace(".md", "")
        formatted_json = parse_document_structure(llm_response, root_name)
        json_file_path = os.path.join(formatted_responses_dir, llm_file.replace('.md', '.json'))
        save_json(json_file_path, formatted_json)

        markdown_lines = format_to_markdown(formatted_json)
        markdown_content = "\n".join(markdown_lines)
        markdown_file_path = os.path.join(markdown_responses_dir, llm_file)
        save_file(markdown_file_path, markdown_content)
