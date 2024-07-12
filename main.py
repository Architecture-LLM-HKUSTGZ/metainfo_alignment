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


def parse_document_structure(text, root_name):      # root_name is the meta_file's name
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


def split_text(text, max_tokens=5000, overlap=100):
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
            api_key='41df71f980554898b556b2ee3d3dc8d1',
            azure_endpoint='https://openai-api-siat.openai.azure.com/',
            api_version='2024-02-01'
        )

    user_prompt = '''
    Please extract the key details from the OCR-processed Method Statement html-format text.
    
    Ensure to retain all specific implementation details, including locations, measurements,
    and other important information, without generalizing. For example, retain details like 
    specific work areas (e.g., 'Works Area W8A, W8B'), measurements, and other critical data points.
    
    Notice:
    - Do not change the section titles. Keep the original section titles as they appear in the OCR-processed html-format text, including their numbering (e.g., '1. **Introduction**', '2. **Scope of Works**').
    - Do not output any extraneous information other than to generate the Method Statement report.
    - The output is in direct markdown format without any explanatory code.
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
        "Method statementforKitchen Small Power and Cable Containment to Depot Kitchen(Builder's Work) wo appendix.md",
        "Predrilling works at CAs 10m away nearest track wo appendix.md",
        "Method Statement for Instrumentation and Monitoring Works at OperationsArea(OA)wo appendix.md",
        "Method Statement for Instrumentation and Monitoring Works alConstruction Area (CA) wo appendix.md",
        "1.13.63 (EDOC) Provision of Electricity Supply to Stoves and Associated Works – ABWF, BS and Modification of existing LV Switchboard Works.md",
        "1.13.61 (EDOC) Lifting Over Test Track for OYB Station.md",
        "1.13.59 (EDOC) Installation of Sheet Piles at Temporary Underfloor Wheel Lathe.md",
        "1.13.58 (EDOC) Cross track duct at West Depot Entry Track and Test Track.md",
        "1.13.55 (CS) Site Entrance, Level Crossing and Cross Road Duct at Test Track for OYB Station Northern Structure, Bifurcation Area and Depot Edge Pile.md",
        "1.13.54 (CS) EDOC for RP Fencing Installation (South Side).md",
        "1.13.52 (EDOC) Mini Pile Installation for OYB Station.md",
        "1.13.50 (CS) General Lifting Plan for North of Bifurcation Works.md",
        "1.13.47 (CS) Method Statement for Seawall Arrangement.md",
        "1.13.46 (BS-SHD-EL) Modification of Busbar in Existing LV Switchboard for New Fuse Switch Panel at Main Depot Building, Siu Ho Wan Depot.md",
        "1.13.45 (CS) Site Clearance and RP Fencing Installation at W2 including OYB Northern Station Structure, Bifurcation Area and Depot Edge Pile.md",
        "1.13.44 (CS) Installation of socket H Pile at bifurcation & OHL mast.md",
        "1.13.42 (EDOC) Installation of Sheet Piles at Temporary Underfloor Wheel Lathe.md",
        "1.13.41 (CS) Installation of Sheet Pile within 10m from nearest track.md",
        "1.13.34 (CS) Method Statement of Disposal of C&D Materials.md"
    ]

    '''Get a list of all text files in the OCR results directory'''
    ocr_files = [f for f in os.listdir(ocr_results_dir) if f.endswith('.md')]

    '''Process each file with a progress bar'''
    for ocr_file in tqdm(ocr_files, desc="Files Aligned"):
        '''单独再对齐一遍文件'''
        if ocr_file not in files_to_reprocess:
            continue

        ocr_file_path = os.path.join(ocr_results_dir, ocr_file)
        documents = read_file(ocr_file_path)
        print(f"{ocr_file} tokens: {token_count(documents)}")
        '''api response'''
        if token_count(documents) > 5000:
            print(f"The documents {ocr_file} tokens exceed 5000, initializing document split!")
            # Split documents if too long
            document_chunks = split_text(documents, max_tokens=5000)

            responses = []
            initial_message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
                {"role": "system", "content": documents},
                {"role": "user", "content": f"The document is too long and has been split into {len(document_chunks)} chunks. I want it processed by {len(document_chunks)} times."},
            ]
            for i, chunk in enumerate(document_chunks):
                messages = initial_message + [
                    {"role": "user", "content": f"""The following is the {i + 1} chunk of {len(document_chunks)}.\n{user_prompt}"""},
                    {"role": "system", "content": chunk}
                ]
                response = document_conversation(client, messages)

                '''keep the script going'''
                if response is None:
                    print(f"The LLM response for {ocr_file} is None!")
                    continue
                responses.append(response)

            # Combine responses
            combined_response = "\n".join(responses)

        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
                {"role": "system", "content": documents}
            ]
            combined_response = document_conversation(client, messages)

            '''keep the script going'''
            if combined_response is None:
                print(f"The LLM response for {ocr_file} is None!")
                continue

        '''saving path'''
        response_saving_path = os.path.join(llm_responses_dir, ocr_file)
        structure_saving_path = os.path.join(parsed_structure_dir, f'{ocr_file.replace(".md", ".json")}')

        '''avoid repetitively processing the same file'''
        """if os.path.exists(response_saving_path) and os.path.exists(structure_saving_path):
            print(f"Skipping {ocr_file} as it has already been processed!")
            continue"""


        '''Save LLM response'''
        save_file(response_saving_path, combined_response)

        '''Convert the parsed structure to JSON format'''
        root_name = ocr_file.replace(".md", "")
        parsed_structure = parse_document_structure(combined_response, root_name)       # Parse the extracted text to get the document structure
        save_json(structure_saving_path, parsed_structure)
