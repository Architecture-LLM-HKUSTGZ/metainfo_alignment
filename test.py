import fitz
from openai import AzureOpenAI
import os
import pytesseract
from PIL import Image
import io
import json
from tqdm import tqdm


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        text += pytesseract.image_to_string(img)
    return text


def parse_document_structure(text, root_name):  # root_name is the meta_file's name
    lines = text.split("\n")
    document_structure = {root_name: {}}
    current_section = None

    for line in lines:
        line = line.strip()
        if line:
            if line.startswith("## ") and '.' in line:
                # New section
                section_title = line.split(" ", 1)[1].strip()
                current_section = section_title
                document_structure[root_name][current_section] = ""
            elif current_section:
                # Append to current section
                document_structure[root_name][current_section] += line + " "

    # Trim trailing spaces from each section content
    for section in document_structure[root_name]:
        document_structure[root_name][section] = document_structure[root_name][section].strip()

    return document_structure


def document_conversation(client, messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=4096,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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


def split_text(text, max_tokens=4500, overlap=100):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(line.split())
        if current_tokens + line_tokens > max_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk = current_chunk[-overlap:] + [line]
            current_tokens = sum(len(ln.split()) for ln in current_chunk)
        else:
            current_chunk.append(line)
            current_tokens += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def token_count(text):
    return len(text.split())


def save_json(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding='utf-8') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    '''
    pdf_path = 'docs/meta_info/1701-W-000-CSC-760-000047.pdf'
    # documents = extract_text_from_pdf(pdf_path)

    # markdown file
    markdown_file = 'docs/examples/1701-W-000-CSC-760-000047.md'
    documents = read_file(markdown_file)
    '''

    client = AzureOpenAI(
        api_key='4568e5656e3f4e9794ec2b6fed460917',
        azure_endpoint='https://openai-hkustgz.openai.azure.com/',
        api_version='2024-02-01'
    )

    user_prompt = '''
    Please extract the key details from the following html-format markdown file.

    Ensure to retain all specific implementation details, including locations, measurements,
    and other important information, without generalizing. For example, retain details like 
    specific work areas (e.g., 'Works Area W8A, W8B'), measurements, and other critical data points.

    Notice:
    - The output is in direct markdown format without any explanatory code. Level-1 headings start with #, Level-2 headings start with ##, Level-3 headings start with ###.
    - Do not change the section titles. Keep the original section titles as they appear in the OCR-processed file, including their numbering and markdown-style format (e.g., '## 1. **<heading>**', '## 2. **<heading>**', etc).
    - Do not output any extraneous information other than to generate the Method Statement report.
    - The first heading of the generated markdown file should be the name of the source file.
    - Do not generate comments like ``markdown``.
    '''

    '''Path of file that needs to be processed'''
    ocr_results_dir = 'results/ocr_results'
    llm_responses_dir = 'results/aligned_files'
    parsed_structure_dir = 'results/parsed_structure'

    '''Ensure the output directories exist'''
    os.makedirs(llm_responses_dir, exist_ok=True)
    os.makedirs(parsed_structure_dir, exist_ok=True)

    # List of specific files to reprocess
    files_to_reprocess = [
        "1.13.71 (CS) Method Statement for Assembly and Disassembly of Crawler Crane at CAs 10m away nearest track (For Model HS8130).md"
    ]

    '''Get a list of all text files in the OCR results directory'''
    ocr_files = [f for f in os.listdir(ocr_results_dir) if f.endswith('.md')]

    '''Process each file with a progress bar'''
    for ocr_file in tqdm(ocr_files, desc="Files Aligned"):

        '''if os.path.exists(os.path.join(llm_responses_dir, ocr_file)) and os.path.exists(os.path.join(parsed_structure_dir, ocr_file)):   # avoid repetitively processing the same file
            print(f"Skipping {ocr_file} as it has already been processed!")
            continue'''

        if ocr_file not in files_to_reprocess:      # 单独再对齐一遍文件
            continue

        ocr_file_path = os.path.join(ocr_results_dir, ocr_file)
        documents = read_file(ocr_file_path)
        print(f"{ocr_file} tokens: {token_count(documents)}")

        '''api response'''
        responses = []
        if token_count(documents) > 4500:
            print(f"The documents {ocr_file} tokens exceed 4500, initializing split!")
            document_chunks = split_text(documents, max_tokens=4500)

            for i, chunk in enumerate(document_chunks):
                if i == 0:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt},
                        {"role": "system", "content": chunk}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "assistant", "content": responses[-1]},
                        {"role": "user", "content": "Please continue."},
                        {"role": "system", "content": chunk}
                    ]

                response = document_conversation(client, messages)
                if response is None:
                    print(f"The API response for {ocr_file} chunk {i + 1} is empty.")
                    continue
                responses.append(response)

            full_response = "\n".join(responses)

        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
                {"role": "system", "content": documents}
            ]
            full_response = document_conversation(client, messages)
            if full_response is None:
                print(f"The API response for {ocr_file} is empty.")

        '''saving path'''
        response_saving_path = os.path.join(llm_responses_dir, ocr_file)
        structure_saving_path = os.path.join(parsed_structure_dir, f'{ocr_file.replace(".md", ".json")}')

        '''Save LLM response'''
        save_file(response_saving_path, full_response)

        '''Convert the parsed structure to JSON format'''
        root_name = ocr_file.replace(".md", "")
        parsed_structure = parse_document_structure(full_response, root_name)       # Parse the extracted text to get the document structure
        save_json(structure_saving_path, parsed_structure)
