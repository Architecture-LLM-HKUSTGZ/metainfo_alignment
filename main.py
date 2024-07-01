import fitz
from openai import AzureOpenAI
import os
import pytesseract
from PIL import Image
import io


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        text += pytesseract.image_to_string(img)
    return text


def document_conversation(client, documents, user_query):
    # 定义会话上下文
    try:
        # 发起聊天完成请求
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": documents},
                {"role": "user", "content": user_query}
            ],
            max_tokens=1024,
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


if __name__ == "__main__":
    pdf_path = 'docs/meta_info/1701-W-000-CSC-760-000047.pdf'
    # documents = extract_text_from_pdf(pdf_path)

    # markdown file
    markdown_file = 'docs/examples/1701-W-000-CSC-760-000047.md'
    documents = read_file(markdown_file)

    ocr_result_path = 'results/ocr_results/1701-W-000-CSC-760-000047.txt'
    response_saving_path = 'results/llm_responses/1701-W-000-CSC-760-000047-md.txt'

    client = AzureOpenAI(
            api_key='41df71f980554898b556b2ee3d3dc8d1',
            azure_endpoint='https://openai-api-siat.openai.azure.com',
            api_version='2024-02-01'
        )

    user_query = "The document is a markdown-format file, what are the main points discussed in the document?"

    response = document_conversation(client, documents, user_query)

    # 保存ocr提取文件与LLM响应文件
    save_file(ocr_result_path, documents)
    save_file(response_saving_path, response)