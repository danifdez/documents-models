
def process_txt(txt_file) -> dict:
    if isinstance(txt_file, str) and (txt_file.startswith('/') or txt_file.startswith('./')):
        with open(txt_file, 'r', encoding='utf-8') as file:
            txt_content = file.read()
    else:
        txt_content = str(txt_file)

    paragraphs = txt_content.split('\n')
    formatted_paragraphs = [f"<p>{p}</p>" for p in paragraphs if p.strip()]
    txt_content = ''.join(formatted_paragraphs)

    return {"content": txt_content}
