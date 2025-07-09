import re
from trafilatura import extract
from bs4 import BeautifulSoup

_author_meta_names = ["author", ":author", "byl", "dc.creator"]
_date_meta_names = [":published_time", ":publishtime", "date", "publication_date","dc.date.issued", "pubdate", "timestamp"]

def _get_meta_content(soup, names):
    meta_tags = soup.find_all("meta", attrs={"content": True})

    for tag in meta_tags:
        for attr in ("name", "property"):
            value = tag.get(attr, "").lower()
            if any(value.endswith(suffix.lower()) for suffix in names):
                return tag["content"]
    return None

def _convert_custom_table_to_html(soup):
    for table in soup.find_all('table'):
        new_table = soup.new_tag('table')
        rows = table.find_all('row')
        for row in rows:
            tr = soup.new_tag('tr')
            cells = row.find_all('cell')
            for cell in cells:
                if cell.get('role') == 'head':
                    th = soup.new_tag('th')
                    th.string = cell.get_text(strip=True)
                    tr.append(th)
                else:
                    td = soup.new_tag('td')
                    td.string = cell.get_text(strip=True)
                    tr.append(td)
            # If no cells, add empty td
            if not cells:
                td = soup.new_tag('td')
                td.string = ''
                tr.append(td)
            new_table.append(tr)
        table.replace_with(new_table)
    return soup

def process_html(html_content) -> dict:
    extracted = extract(
        html_content, output_format="html", favor_precision=True, include_formatting=True, include_links=True, include_images=True, include_tables=True)

    soup = BeautifulSoup(html_content, "html.parser")
    title = soup.title.string if soup.title else "No Title"
    author = _get_meta_content(soup, _author_meta_names)
    publication_date = _get_meta_content(soup, _date_meta_names)

    if extracted:
        try:
            clean_soup = BeautifulSoup(extracted, "html.parser")

            is_fragment = not clean_soup.find('html')

            for tag in clean_soup.find_all(True):
                if tag.has_attr('style'):
                    del tag['style']
                if tag.has_attr('class'):
                    del tag['class']
                if tag.has_attr('id'):
                    del tag['id']

            for div in clean_soup.find_all('div'):
                if not div.find(['div', 'p', 'ul', 'ol', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    div.name = 'p'

            if is_fragment:
                top_elements = [
                    el for el in clean_soup.children if el.name is not None]

                if len(top_elements) == 1 and top_elements[0].name == 'div':
                    top_elements[0].unwrap()

            else:
                body = clean_soup.find('body')
                if body:
                    body_elements = [
                        el for el in body.children if el.name is not None]

                    if len(body_elements) == 1 and body_elements[0].name == 'div':
                        body_elements[0].unwrap()

            divs_to_check = list(clean_soup.find_all('div'))
            for div in divs_to_check:
                if not div.attrs and len([c for c in div.children if c.name == 'div']) == 1 and len([c for c in div.children if c.name is not None]) == 1:
                    div.unwrap()

            for div in clean_soup.find_all('div'):
                div.unwrap()

            if not is_fragment:
                body = clean_soup.find('body')
                if body:
                    body_html = ''.join(str(content)
                                        for content in body.contents)
                    clean_soup = BeautifulSoup(body_html, 'html.parser')
                else:
                    html_tag = clean_soup.find('html')
                    if html_tag:
                        head = clean_soup.find('head')
                        if head:
                            head.decompose()

            for tag in clean_soup.find_all(['head', 'script', 'style']):
                tag.decompose()

            for graphic in clean_soup.find_all('graphic'):
                graphic.name = 'img'

            clean_html = str(clean_soup).strip()

            # Convert custom tables to standard HTML tables
            clean_soup = _convert_custom_table_to_html(clean_soup)
            clean_html = str(clean_soup)

            clean_html = re.sub(r'\s+', ' ', clean_html)
            clean_html = re.sub(r'<p>\s*</p>', '', clean_html)
            clean_html = re.sub(
                r'<html>|</html>|<body>|</body>', '', clean_html)

            extracted = clean_html.strip()
        except Exception as e:
            print(f"Error processing HTML content: {str(e)}")

    if extracted:
        extracted = extracted.strip()

    return {"content": extracted, "title": title, "author": author, "publication_date": publication_date}
