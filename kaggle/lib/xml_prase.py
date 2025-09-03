import re
import os
import time
import unicodedata
from typing import List, Tuple, Optional, Dict
import xml.etree.ElementTree as ET
from lxml import etree
def fix_broken_links(text: str) -> str:
    patterns = [
        (r'https?://\s*([a-zA-Z0-9\.\-]+)', r'https://\1'),
        (r'(doi\.)\s+(org/10\.)', r'\1\2'),
        (r'doi\.org/\s+10\.', r'doi.org/10.'),
        (r'doi\.org/10\.\s*(\d+)', r'doi.org/10.\1'),
        (r'doi\.org/10\.(\d{4,5})/\s+([a-zA-Z0-9\.\-_]+)', r'doi.org/10.\1/\2'),
        # (r'(https?://[a-zA-Z0-9\.\-/]+/)\s+([a-zA-Z0-9\.\-_/]+)', r'\1\2'),
        # (r'(https?://[a-zA-Z0-9\.\-/]+-)\s+([a-zA-Z0-9\.\-_/]+)', r'\1\2'),
        (r'(10\.\d{4,5}/[a-zA-Z0-9\.\-/]+\.)\s+([a-z0-9\.\-_/]{4,20},?\s+|(?!(?:DATAS?|RESULTS?|DISCUSSIONS?|REFERENCES?|CONFLICT|ORCID|ACKNOWLEDGMENTS?)\b)[A-Z0-9\.\-_/]{4,20}\s+)', r'\1\2'),
        # (r'(https?://[^\s]*[?&])\s+([a-zA-Z0-9=&\.\-_]+)', r'\1\2'),
        (r'\n+(https?://doi\.org/[^\s]*)', r'\1'),
        (r'(\(.{0,20}10\.\d{4,}/.{1,30})\s+([A-Za-z0-9]{1,10}\))', r'\1\2'),
        (r'(10\.\d{4,5}/[a-zA-Z0-9\.\-/]+-)\s+([a-zA-Z0-9\.\-_/]+)', r'\1\2'),
        (r'(doi\.org/10\.\d{4,5}/[a-zA-Z0-9\.\-/]+-)\s+([a-zA-Z0-9\.\-_/]+)', r'\1\2'),
        (r'(https://doi\.org/10\.\d{4,5}/[a-zA-Z0-9\.\-/]+-)\s+([a-zA-Z0-9\.\-_/]+)', r'\1\2'),
        (r'(10\.\d{4,5}/[a-zA-Z0-9\.\-/]+-)\s+([a-zA-Z0-9\.\-_/]+)', r'\1\2'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    
    return text
def text_clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", text)
    text = re.sub(r'<br>', '', text)
    text = re.sub(r'\n&', '&', text)
    text = re.sub(r'(\d+\.) +(\d+)', r'\1\2', text)
    text = re.sub(r'/\s+', '/', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\\_', '_', text)
    text = re.sub(r'\-\n', '-', text)
    text = re.sub(r'\u00ad\s?', '', text)
    
    return text
# def xml_kind(xml_path: str) -> str:
#     try:
#         with open(xml_path, 'rb') as f:
#             head = f.read(2048).decode('utf8', 'ignore')
#         if 'www.tei-c.org/ns' in head:
#             return 'tei'
#         if re.search(r'(NLM|TaxonX)//DTD', head):
#             return 'jats'
#         if 'www.wiley.com/namespaces' in head:
#             return 'wiley'
#         if 'BioC.dtd' in head:
#             return 'bioc'
#         return 'unknown'
#     except Exception:
#         return 'unknown'
def robust_xml_parse(file_path: str, timeout: int = 120) -> Tuple[Optional[ET.Element], Dict]:
    start_time = time.time()
    parse_info = {
        'success': False,
        'encoding': None,
        'parse_time': 0,
        'error': None,
        'method': None
    }
    try:
        if time.time() - start_time > timeout:
            raise TimeoutError("解析超时")
        tree = ET.parse(file_path)
        root = tree.getroot()
        parse_info.update({
            'success': True,
            'encoding': 'utf-8',
            'parse_time': time.time() - start_time,
            'method': 'standard'
        })
        return root, parse_info
    except Exception as e:
        parse_info['error'] = str(e)
    try:
        if time.time() - start_time > timeout:
            raise TimeoutError("解析超时")
        
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        
        root = ET.fromstring(content)
        parse_info.update({
            'success': True,
            'encoding': encoding,
            'parse_time': time.time() - start_time,
            'method': 'encoding_detection'
        })
        return root, parse_info
    except Exception as e:
        parse_info['error'] = str(e)
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            if time.time() - start_time > timeout:
                raise TimeoutError("解析超时")
            
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            
            root = ET.fromstring(content)
            parse_info.update({
                'success': True,
                'encoding': encoding,
                'parse_time': time.time() - start_time,
                'method': f'multi_encoding_{encoding}'
            })
            return root, parse_info
        except Exception as e:
            continue
    try:
        if time.time() - start_time > timeout:
            raise TimeoutError("解析超时")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        content = clean_xml_content(content)
        root = ET.fromstring(content)
        parse_info.update({
            'success': True,
            'encoding': 'utf-8',
            'parse_time': time.time() - start_time,
            'method': 'cleaned_content'
        })
        return root, parse_info
    except Exception as e:
        parse_info['error'] = str(e)
    try:
        if time.time() - start_time > timeout:
            raise TimeoutError("解析超时")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(500000)  # 读取前500KB
        last_complete = content.rfind('>')
        if last_complete > 0:
            content = content[:last_complete + 1]
        if not content.strip().startswith('<'):
            content = f'<root>{content}</root>'
        root = ET.fromstring(content)
        parse_info.update({
            'success': True,
            'encoding': 'utf-8',
            'parse_time': time.time() - start_time,
            'method': 'chunked_parsing'
        })
        return root, parse_info
    except Exception as e:
        parse_info['error'] = str(e)
    
    parse_info['parse_time'] = time.time() - start_time
    return None, parse_info
# def extract_text(root, article_id: str, verbose: bool = False) -> List[Tuple[str, str, str, str]]:
#     try:
#         all_text = ' '.join(root.itertext())
#         import unicodedata
#         all_text = unicodedata.normalize('NFKC', all_text)
#         all_text = text_clean(all_text)
#         all_text = re.sub(r'[^\x00-\x7F]', '', all_text)
#         all_text = fix_broken_links(all_text)
#         # all_text = re.sub(r'\s+', ' ', all_text).strip()
#                 # 保留段落结构，只清理多余的空白
#         # all_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', all_text)  # 多个空行变为两个
#         # all_text = re.sub(r'[ \t]+', ' ', all_text)  # 多个空格/制表符变为一个
#         # all_text = re.sub(r'\n[ \t]+', '\n', all_text)  # 行首空白
#         # all_text = re.sub(r'[ \t]+\n', '\n', all_text)  # 行尾空白
#         return all_text.strip()
#     except Exception as e:
#         print(f"XML解析错误 {xml_path}: {e}")
#         return ''
# def xml2text(xml_path: str) -> str:
#     try:
#         with open(xml_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#         # text_only = re.sub(r'<[^>]+>', ' ', content)
#         # # text_only = re.sub(r'\s+', ' ', text_only)
#         # text_only = re.sub(r'&[a-zA-Z]+;', ' ', text_only)
#         # text_only = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text_only)
#         # text_only = re.sub(r'&#[0-9]+;', ' ', text_only)
#         text_only = re.sub(r'(10\.\d{4,9}/)\s+', r'\1', text_only)
#         if text_only:
            
#             text_only = unicodedata.normalize('NFKC', text_only)
#             text_only = text_clean(text_only)
#             text_only = re.sub(r'[^\x00-\x7F]', '', text_only)
#             text_only = fix_broken_links(text_only)
#             # text_only = re.sub(r'\s+', ' ', text_only)
#                     # 保留段落结构，只清理多余的空白
#             # text_only = re.sub(r'\n\s*\n\s*\n+', '\n\n', text_only)  # 多个空行变为两个
#             # text_only = re.sub(r'[ \t]+', ' ', text_only)  # 多个空格/制表符变为一个
#             # text_only = re.sub(r'\n[ \t]+', '\n', text_only)  # 行首空白
#             # text_only = re.sub(r'[ \t]+\n', '\n', text_only)  # 行尾空白
#             text_only = text_only.strip()
        
#         return text_only
        
#     except Exception as e:
#         print(f"XML解析错误 {xml_path}: {e}")
#         return ""
# def extract_references_from_xml_enhanced(xml_path: str, timeout: int = 30, verbose: bool = False) -> List[Tuple[str, str, str, str]]:
#     filename = os.path.basename(xml_path)
#     article_id = filename.split(".xml")[0]
#     xml_type = xml_kind(xml_path)
#     root, parse_info = robust_xml_parse(xml_path, timeout)
#     if not root:
#         return ''
#     full_text = extract_text(root, article_id, verbose)
#     if not full_text:
#         full_text = xml2text(xml_path)
#     return full_text


def xml_kind(xml_path: str) -> str:
    """
    判断XML文件的类型 (TEI, JATS, Wiley, BioC, unknown)。
    """
    try:
        with open(xml_path, 'rb') as f:
            head = f.read(2048).decode('utf8', 'ignore')
        if 'www.tei-c.org/ns' in head:
            return 'tei'
        if re.search(r'(NLM|TaxonX)//DTD', head):
            return 'jats'
        if 'www.wiley.com/namespaces' in head:
            return 'wiley'
        if 'BioC.dtd' in head:
            return 'bioc'
        return 'unknown'
    except Exception as e:
        # 可选：记录日志或打印错误
        # print(f"Error determining XML kind for {xml_path}: {e}")
        return 'unknown' # 或者返回 None，取决于调用方如何处理
def clean_xml_content(content: str) -> str:
    content = re.sub(r'10\.\d{4,9}/\s+', '10.', content)
    content = re.sub(r'(10\.\d{4,9})\s*/\s*', r'\1/', content)
    return content


def xml2text(xml_path: str) -> str:
    """
    根据XML类型分类，优化提取文本内容，提高可读性。
    """
    try:
        # 1. 判断XML类型
        kind = xml_kind(xml_path)

        # 2. 使用健壮的解析器
        root, parse_info = robust_xml_parse(xml_path)
        if root is None:
            print(f"Failed to parse XML {xml_path}. Info: {parse_info}")
            return ""

        # 3. 根据类型提取文本
        txt_parts = [] # 使用列表存储不同部分，最后再 join
        namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'} # 定义 TEI 命名空间

        if kind == 'tei':
            # 尝试查找 TEI 标准结构
            # 查找 <text> 或 <body>
            text_elem = root.find('.//tei:text', namespaces) or root.find('.//text') # 带命名空间或不带
            body_elem = root.find('.//tei:body', namespaces) or root.find('.//body')
            back_elem = root.find('.//tei:listbibl', namespaces) or root.find('.//listbibl')

            # --- 处理正文部分 (body) ---
            content_elem = body_elem if body_elem is not None else text_elem
            if content_elem is not None:
                for elem in content_elem.iter(): # 遍历 body (或 text) 下的所有元素
                    # --- 关键优化：跳过属于 <back> 子树的元素 ---
                    if back_elem is not None and elem in back_elem.iter():
                        # print(f"Skipping {elem.tag} as it's inside <back>/<listbibl>") # Debug
                        continue # 跳过整个 <back> 子树

                    # --- 处理非 <back> 内的元素 ---
                    tag_local_name = elem.tag.split('}')[-1]

                    # 处理常见的块级元素
                    block_tags = {'p', 'head', 'list', 'item', 'figure', 'table', 'lg', 'l'}
                    if tag_local_name in block_tags:
                        block_text = ''.join(elem.itertext()).strip()
                        if block_text:
                            block_text = re.sub(r'[ \t\r\f\v]+', ' ', block_text)
                            txt_parts.append(block_text)
                            txt_parts.append('\n')

                    # 处理在 <body> 内但不在 <back> 内的 <biblstruct>
                    # (例如，脚注或正文中的引用)
                    elif tag_local_name == 'biblstruct':
                        # print(f"Found <biblstruct> in body (outside back): {elem.get('xml:id', 'no-id')}") # Debug
                        bibl_text = _extract_biblstruct_text(elem)
                        if bibl_text:
                            txt_parts.append(bibl_text)
                            txt_parts.append('\n')

                    # 处理 <lb/> (换行) 标签 (TEI中常见，但在块级元素处理中较难精确插入)
                    # 这里简化处理，主要依赖块级元素换行

            # --- 处理参考文献部分 (<back>) ---
            if back_elem is not None:
                # 在 <back> 中也可能有 <biblstruct>
                # 可以查找 <div type="references"> 或直接查找所有 <biblstruct>
                ref_divs = back_elem.findall('.//tei:div[@type="references"]', namespaces) or \
                            back_elem.findall('.//div[@type="references"]')
                ref_list = []
                if ref_divs:
                    # 如果有专门的 references div
                    for ref_div in ref_divs:
                        biblstructs = ref_div.findall('.//tei:biblstruct', namespaces) or \
                                        ref_div.findall('.//biblstruct')
                        for bibl_elem in biblstructs:
                            bibl_text = _extract_biblstruct_text(bibl_elem)
                            if bibl_text:
                                ref_list.append('\n')
                                ref_list.append(bibl_text)
                else:
                    # 直接在 <back> 下查找 <biblstruct>
                    biblstructs = back_elem.findall('.//tei:biblstruct', namespaces) or \
                                    back_elem.findall('.//biblstruct')
                    for bibl_elem in biblstructs:
                        bibl_text = _extract_biblstruct_text(bibl_elem)
                        if bibl_text:
                            ref_list.append('\n')
                            ref_list.append(bibl_text)

                # 如果找到了参考文献条目
                if ref_list:
                    # print('15158484')
                    txt_parts.append('\n') # 与正文分开
                    txt_parts.append("References") # 添加标题
                    txt_parts.append('\n')
                    # txt_parts.extend(ref_list) # 直接添加条目列表
                    # 或者，像 format_references 一样，处理每个条目并添加句号
                    formatted_ref_entries = []
                    for entry_text in ref_list:
                        #  entry_text = entry_text.strip()
                            # 修复作者名的单个字母（添加点） - 在 _extract_biblstruct_text 中可能已部分处理，
                            # 但在这里再做一遍更保险，以防条目结构不完全标准
                            entry_text = re.sub(r'\b([A-Z])\s+([A-Z][a-z]+)', r'\1. \2', entry_text)
                            entry_text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z][a-z]+)', r'\1. \2. \3', entry_text)
                            if entry_text and entry_text != "\n" and not entry_text.endswith('.'):
                                entry_text += '.'
                            formatted_ref_entries.append(entry_text)
                    txt_parts.extend(formatted_ref_entries) # 添加格式化后的条目
                    # References 部分条目间通常也需要换行
                    # 但我们是逐个添加的，所以这里不需要额外 join('\n')

            # 如果连 <text> 或 <body> 都找不到 (不太可能)，则回退
            if content_elem is None and back_elem is None:
                print(f"Warning: Could not find <text>/<body> or <back> in TEI file {xml_path}. Falling back to full text extraction.")
                fallback_text = ' '.join(root.itertext())
                fallback_text = re.sub(r'[ \t\r\f\v]+', ' ', fallback_text) # 规范化
                txt_parts.append(fallback_text)

        elif kind == 'bioc':
            # --- 针对 BioC 格式的优化处理 (支持元数据提取) ---
            # BioC 结构通常是 <collection> -> <document> -> <passage> -> <text>
            # 优化目标：提取 <passage> 下的 <text> 内容。
            # 特殊处理：如果 <passage> 包含参考文献元数据 (<infon key="section_type">REF</infon>)，
            #           则提取 DOI 和作者名，并与 <text> 合并。

            # 查找所有 <document> 元素
            documents = root.findall('.//document')
            if not documents:
                # 如果顶层就是 <document> (不太常见，但兼容一下)
                documents = [root] if root.tag == 'document' else []

            # 用于存储提取到的文本片段
            document_parts = [] # 用于存储每个 document 的内容，方便文档间分隔

            for doc_elem in documents:
                doc_parts = [] # 存储当前 document 的所有 passage 内容
                # 在每个 <document> 中查找 <passage> 元素
                passages = doc_elem.findall('.//passage')
                if not passages:
                    # 如果 <document> 下直接是 <passage> (结构变体)
                    if doc_elem.tag == 'passage':
                        passages = [doc_elem]
                    # 或者 <document> 下直接是 <text> (更简化的结构)
                    elif doc_elem.find('text') is not None:
                        passages = [doc_elem] # Treat document as a single passage

                passage_processed = False
                for passage_elem in passages:
                    passage_text_parts = [] # 存储当前 passage 的最终文本（可能由多部分组成）

                    # --- 特殊处理：检查是否为参考文献条目 ---
                    section_type_elem = passage_elem.find('./infon[@key="section_type"]')
                    is_reference = section_type_elem is not None and section_type_elem.text == 'REF'

                    text_elem = passage_elem.find('text')
                    base_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""

                    if is_reference and base_text:
                        # --- 提取参考文献元数据 ---
                        ref_parts = []

                        # 1. 提取 DOI
                        doi_elem = passage_elem.find('./infon[@key="pub-id_doi"]')
                        if doi_elem is not None and doi_elem.text:
                             doi_text = doi_elem.text.strip()
                             if doi_text:
                                 # 格式化为标准 DOI 链接
                                 if not doi_text.startswith("http"):
                                     formatted_doi = f"https://doi.org/{doi_text}"
                                 else:
                                     formatted_doi = doi_text
                                 ref_parts.append(formatted_doi)

                        # 2. 提取作者 (从 name_0, name_1, ... infon 中)
                        author_names = []
                        infon_index = 0
                        while True:
                            author_infon_key = f"name_{infon_index}"
                            author_infon_elem = passage_elem.find(f'./infon[@key="{author_infon_key}"]')
                            if author_infon_elem is not None and author_infon_elem.text:
                                # infon 内容格式通常是 "surname:...;given-names:..."
                                author_info = author_infon_elem.text.strip()
                                # 简单解析：按 ';' 分割 surname 和 given-names 部分
                                parts = author_info.split(';')
                                surname = ""
                                given_names = ""
                                for part in parts:
                                    if part.startswith("surname:"):
                                        surname = part[len("surname:"):].strip()
                                    elif part.startswith("given-names:"):
                                        given_names = part[len("given-names:"):].strip()
                                # 拼接格式：Given Names Surname
                                if surname or given_names:
                                    full_name = f"{given_names} {surname}".strip()
                                    author_names.append(full_name)
                                infon_index += 1
                            else:
                                break # 没有更多 name_* infon 了

                        if author_names:
                            authors_str = ', '.join(author_names)
                            ref_parts.append(authors_str)

                        # 3. 添加原始文本 (通常是标题)
                        if base_text:
                            # 规范化基础文本的空格
                            norm_base_text = re.sub(r'[ \t\r\f\v]+', ' ', base_text)
                            ref_parts.append(norm_base_text)

                        # 4. 合并参考文献条目各部分 (用 '. ' 分隔，符合常见引用格式)
                        if ref_parts:
                            combined_ref_text = '. '.join(ref_parts)
                            # 确保条目以句号结尾
                            if not combined_ref_text.endswith('.'):
                                combined_ref_text += '.'
                            passage_text_parts.append(combined_ref_text)

                    elif base_text: # 非参考文献或没有基础文本则按原样处理
                        # --- 标准处理：提取并规范化普通文本 ---
                        norm_text = re.sub(r'[ \t\r\f\v]+', ' ', base_text)
                        passage_text_parts.append(norm_text)

                    # --- 将处理好的 passage 文本添加到当前文档 ---
                    if passage_text_parts:
                        # 使用换行符连接 passage 内的各个部分（虽然通常只有一个部分）
                        final_passage_text = '\n'.join(passage_text_parts).strip()
                        if final_passage_text:
                            doc_parts.append(final_passage_text)
                            doc_parts.append('\n') # 每个 passage 后换行
                            passage_processed = True

                # --- 将处理好的 document 内容添加到主列表 ---
                if doc_parts:
                    # 移除最后可能多余的换行
                    if doc_parts and doc_parts[-1] == '\n':
                        doc_parts.pop()
                    document_parts.append(doc_parts)
                    # 如果处理了内容，且有多个文档，可以考虑在文档间加分隔符
                    # if passage_processed and len(documents) > 1:
                    #     document_parts.append(['\n---\n']) # 可选：文档间分隔符

            # --- 将所有文档内容合并到最终的 txt_parts ---
            if document_parts:
                 for i, doc_parts in enumerate(document_parts):
                      txt_parts.extend(doc_parts)
                      # 在文档之间添加分隔（如果需要且不是最后一个文档）
                      # if i < len(document_parts) - 1:
                      #     txt_parts.append('\n---\n')
            else:
                 # 如果完全没有找到符合 BioC 结构的内容，回退到提取所有文本
                 print(f"Warning: Could not find standard BioC structure in {xml_path}. Falling back to full text extraction.")
                 fallback_text = ' '.join(root.itertext())
                 fallback_text = re.sub(r'[ \t\r\f\v]+', ' ', fallback_text) # 规范化
                 txt_parts.append(fallback_text)
        elif kind == 'jats':
            # --- 针对 JATS 格式的优化处理 ---
            # JATS 结构通常是 ... <body>...<p>...</p>...</body> ... <back>...<ref-list>...<ref>...</ref>...</ref-list>...</back>
            # 目标：提取 <body> 内的段落，并提取 <back> -> <ref-list> -> <ref> 内的参考文献

            # 1. 查找 <body> 元素 (正文)
            body_elem = root.find('.//body')

            # 2. 查找 <back> 元素 (包含参考文献等)
            back_elem = root.find('.//back')

            # 3. 查找 <ref-list> 元素 (参考文献列表)，通常在 <back> 内
            # 优先在 <back> 内查找，如果没有 <back>，则在整个文档中查找
            ref_list_elem = None
            if back_elem is not None:
                ref_list_elem = back_elem.find('.//ref-list')
            if ref_list_elem is None:
                ref_list_elem = root.find('.//ref-list') # Fallback

            # --- 处理正文部分 (body) ---
            if body_elem is not None:
                # 遍历 <body> 下的所有元素，提取段落等块级内容
                # 常见的 JATS 块级元素标签
                direct_block_tags = {'p', 'title', 'list', 'disp-formula', 'fig', 'table-wrap', 'statement', 'boxed-text', 'verse-group'}
                # 'sec' 是结构性的，通常包含其他块
                container_tags = {'sec'} 

                def process_element(elem: ET.Element, is_within_container: bool = False):
                    """递归处理元素，避免重复提取嵌套块内容"""
                    tag_local_name = elem.tag 

                    # 1. 检查是否为直接块级元素
                    if tag_local_name in direct_block_tags:
                        block_text = ' '.join(elem.itertext()).strip()
                        if block_text:
                            block_text = re.sub(r'[ \t\r\f\v]+', ' ', block_text)
                            txt_parts.append(block_text)
                            txt_parts.append('\n')
                        # 处理完直接块后，通常不需要再遍历其子元素以避免重复
                        # (除非子元素有特殊需要，但这里简化处理)
                        # 因此，我们不递归处理 elem 的子元素
                        return # 关键：处理完直接块就返回，不处理子元素

                    # 2. 检查是否为容器元素 (如 <sec>)
                    elif tag_local_name in container_tags:
                        # 对于容器，我们不直接提取其文本，而是处理其直接子元素
                        # 这样可以找到 <sec> 下的 <p>, <title> 等块
                        for child in elem:
                            process_element(child, is_within_container=True)
                        return # 处理完容器的子元素就返回

                    # 3. 如果不是块也不是容器，但我们在容器内部，则检查其子元素
                    # 这是为了处理 <sec> -> <something_not_block> -> <p> 这种情况
                    # 或者处理顶层非块元素（虽然不常见）
                    elif is_within_container:
                            # 递归检查子元素
                            for child in elem:
                                process_element(child, is_within_container=True)
                    else:
                        # 顶层非块非容器元素，通常忽略或只检查其子元素
                        # 例如，<body> 下可能直接有 <p> (已处理) 或 <sec> (已处理)
                        # 或者一些元数据标签，我们通常不提取。
                        # 为安全起见，如果顶层 <body> 下有非标准块，也检查其子元素
                        for child in elem:
                            process_element(child, is_within_container=False)

                # --- 启动处理 ---
                # 遍历 <body> 的直接子元素
                for child_elem in body_elem:
                    process_element(child_elem, is_within_container=False)

                    # 注意：JATS 中的参考文献通常在 <back>/<ref-list>/<ref> 中，
                    # 而不是直接在 <body> 里作为 <ref>。但如果存在，也需要跳过。
                    # 我们将在处理 ref-list 时专门提取。
                ack_elem = back_elem.find('.//ack')
                if ack_elem is not None:
                    ack_title_elem = ack_elem.find('./title')
                    ack_title = "Acknowledgements"
                    if ack_title_elem is not None and ack_title_elem.text:
                        ack_title = ack_title_elem.text.strip()
                    ack_content = ''.join(ack_elem.itertext()).strip()
                    # 移除标题文本本身（如果它被 itertext 包含了）
                    if ack_title_elem is not None and ack_title_elem.text:
                            ack_title_text = ack_title_elem.text.strip()
                            ack_content = ack_content[len(ack_title_text):].strip()
                    if ack_content:
                        txt_parts.append('\n') # 与前面内容分开
                        txt_parts.append(ack_title)
                        txt_parts.append('\n')
                        # 规范化 Ack 内容 (保留段落结构)
                        ack_content = re.sub(r'[ \t\r\f\v]+', ' ', ack_content) # 规范化空格
                        # ack_content = re.sub(r'([.!?])\s*', r'\1\n', ack_content) # 句号后换行（简单分句）
                        ack_content = re.sub(r'\n\s*\n+', '\n\n', ack_content) # 限制空行
                        txt_parts.append(ack_content.strip())
                        txt_parts.append('\n')

                # 3. 提取 Author Contributions (<notes notes-type="author-contribution">)
                author_contrib_elem = back_elem.find('.//notes[@notes-type="author-contribution"]')
                if author_contrib_elem is not None:
                    contrib_title_elem = author_contrib_elem.find('./title')
                    contrib_title = "Author contributions"
                    if contrib_title_elem is not None and contrib_title_elem.text:
                        contrib_title = contrib_title_elem.text.strip()
                    contrib_content = ''.join(author_contrib_elem.itertext()).strip()
                    if contrib_title_elem is not None and contrib_title_elem.text:
                            contrib_title_text = contrib_title_elem.text.strip()
                            contrib_content = contrib_content[len(contrib_title_text):].strip()
                    if contrib_content:
                        txt_parts.append('\n')
                        txt_parts.append(contrib_title)
                        txt_parts.append('\n')
                        contrib_content = re.sub(r'[ \t\r\f\v]+', ' ', contrib_content)
                        # contrib_content = re.sub(r'([.!?])\s*', r'\1\n', contrib_content)
                        contrib_content = re.sub(r'\n\s*\n+', '\n\n', contrib_content)
                        txt_parts.append(contrib_content.strip())
                        txt_parts.append('\n')

                # 4. 提取 Data Availability (<notes notes-type="data-availability">)
                data_avail_elem = back_elem.find('.//notes[@notes-type="data-availability"]')
                if data_avail_elem is not None:
                    data_title_elem = data_avail_elem.find('./title')
                    data_title = "Data availability"
                    if data_title_elem is not None and data_title_elem.text:
                        data_title = data_title_elem.text.strip()
                    data_content = ''.join(data_avail_elem.itertext()).strip()
                    if data_title_elem is not None and data_title_elem.text:
                            data_title_text = data_title_elem.text.strip()
                            data_content = data_content[len(data_title_text):].strip()
                    if data_content:
                        txt_parts.append('\n')
                        txt_parts.append(data_title)
                        txt_parts.append('\n')
                        data_content = re.sub(r'[ \t\r\f\v]+', ' ', data_content)
                        # data_content = re.sub(r'([.!?])\s*', r'\1\n', data_content)
                        data_content = re.sub(r'\n\s*\n+', '\n\n', data_content)
                        txt_parts.append(data_content.strip())
                        txt_parts.append('\n')

                # 5. 提取 Competing Interests / Conflict of Interest (<notes notes-type="COI-statement">)
                coi_elem = back_elem.find('.//notes[@notes-type="COI-statement"]')
                if coi_elem is not None:
                        # 也可能使用 <notes notes-type="conflict-of-interest"> 或直接在 <back> 下找 <sec> 或 <p> 包含 "conflict" 的
                        # 这里只处理明确的 COI-statement
                    coi_title_elem = coi_elem.find('./title')
                    coi_title = "Competing interests"
                    if coi_title_elem is not None and coi_title_elem.text:
                        coi_title = coi_title_elem.text.strip()
                    coi_content = ''.join(coi_elem.itertext()).strip()
                    if coi_title_elem is not None and coi_title_elem.text:
                            coi_title_text = coi_title_elem.text.strip()
                            coi_content = coi_content[len(coi_title_text):].strip()
                    if coi_content:
                        txt_parts.append('\n')
                        txt_parts.append(coi_title)
                        txt_parts.append('\n')
                        coi_content = re.sub(r'[ \t\r\f\v]+', ' ', coi_content)
                        # coi_content = re.sub(r'([.!?])\s*', r'\1\n', coi_content)
                        coi_content = re.sub(r'\n\s*\n+', '\n\n', coi_content)
                        txt_parts.append(coi_content.strip())
                        txt_parts.append('\n')

                # 6. 提取 Supplementary Information (<sec> with <title>Supplementary information</title>)
                # 查找所有 <sec>，检查其 <title>
                sec_elems = back_elem.findall('.//sec')
                for sec_elem in sec_elems:
                    sec_title_elem = sec_elem.find('./title')
                    if sec_title_elem is not None and "supplementary" in sec_title_elem.text.strip().lower():
                        sec_title = sec_title_elem.text.strip()
                        sec_content = ''.join(sec_elem.itertext()).strip()
                        if sec_title_elem is not None and sec_title_elem.text:
                            sec_title_text = sec_title_elem.text.strip()
                            sec_content = sec_content[len(sec_title_text):].strip()
                        if sec_content:
                            txt_parts.append('\n')
                            txt_parts.append(sec_title)
                            txt_parts.append('\n')
                            sec_content = re.sub(r'[ \t\r\f\v]+', ' ', sec_content)
                            # sec_content = re.sub(r'([.!?])\s*', r'\1\n', sec_content)
                            sec_content = re.sub(r'\n\s*\n+', '\n\n', sec_content)
                            txt_parts.append(sec_content.strip())
                            txt_parts.append('\n')
                        break # 通常只有一个 Supplementary Information 部分

                # 7. 提取 Publisher's Note / Footnotes (<fn-group>)
                fn_group_elem = back_elem.find('.//fn-group')
                if fn_group_elem is not None:
                    fn_title_elem = fn_group_elem.find('./title') # 可能有标题
                    fn_title = "Footnotes" # 默认标题
                    if fn_title_elem is not None and fn_title_elem.text:
                        fn_title = fn_title_elem.text.strip()

                    fn_elems = fn_group_elem.findall('.//fn')
                    if fn_elems:
                        txt_parts.append('\n')
                        txt_parts.append(fn_title)
                        txt_parts.append('\n')
                        fn_texts = []
                        for fn_elem in fn_elems:
                            fn_text = ''.join(fn_elem.itertext()).strip()
                            if fn_text:
                                fn_text = re.sub(r'[ \t\r\f\v]+', ' ', fn_text)
                                fn_texts.append(fn_text)
                        if fn_texts:
                            # 用换行连接脚注
                            txt_parts.append('\n'.join(fn_texts))
                            txt_parts.append('\n')

            # --- 处理参考文献部分 (<back> -> <ref-list> -> <ref>) ---
            # 目标：找到 <ref-list>，然后提取其中每个 <ref> 的文本作为独立条目
            if ref_list_elem is not None:
                ref_entries = ref_list_elem.findall('.//ref')
                ref_texts = []
                for ref_elem in ref_entries:
                    # 提取 <ref> 元素内的所有文本，并合并为单行
                    # <ref> 内可能有复杂的标签，如 <mixed-citation>, <element-citation> 等
                    # 为了简化，我们提取所有文本并规范化
                    ref_text = ' '.join(ref_elem.itertext()).strip()
                    if ref_text:
                        # 规范化参考文献条目内的空白 (合并为单行)
                        ref_text = re.sub(r'\s+', ' ', ref_text)
                        ref_texts.append(ref_text)
                        ref_texts.append('\n')


                # 如果找到了参考文献条目
                if ref_texts:
                    txt_parts.append('\n') # 与正文分开
                    txt_parts.append('References')
                    txt_parts.append('\n')
                    # 检查是否已有 "References" 或 "Bibliography" 标题
                    has_title = False
                    # 简单检查 ref-list 或 back 是否有标题
                    ref_list_title = ref_list_elem.find('./title')
                    back_title = back_elem.find('./title') if back_elem is not None else None
                    if ref_list_title is not None and ref_list_title.text:
                        title_text = ref_list_title.text.strip().lower()
                        if title_text in ('references', 'bibliography', '参考文献'):
                            has_title = True
                    elif back_title is not None and back_title.text:
                        title_text = back_title.text.strip().lower()
                        if title_text in ('references', 'bibliography', '参考文献'):
                            has_title = True

                    if not has_title:
                        txt_parts.append("References") # 添加标准标题
                        txt_parts.append('\n')

                    # 格式化并添加参考文献条目
                    # (类似于 TEI 和你之前验证正确的逻辑)
                    formatted_ref_entries = []
                    for entry_text in ref_texts:
                        entry_text = entry_text
                        # 修复作者名缩写 (JATS 条目也可能需要)
                        entry_text = re.sub(r'\b([A-Z])\s+([A-Z][a-z]+)', r'\1. \2', entry_text)
                        entry_text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z][a-z]+)', r'\1. \2. \3', entry_text)
                        # 确保以句号结尾 (JATS 条目末尾可能没有标准句号)
                        if entry_text and entry_text != "\n" and not entry_text.endswith('.'):
                            entry_text += '.'
                        formatted_ref_entries.append(entry_text)

                    txt_parts.extend(formatted_ref_entries)
                    # txt_parts.append('\n') # 最后一个条目后通常不需要额外换行

            # Fallback: 如果没有找到 <body> 也没有 <ref-list>，则回退
            if body_elem is None and ref_list_elem is None:
                print(f"Warning: Could not find <body> or <ref-list> in JATS file {xml_path}. Falling back to full text extraction.")
                fallback_text = ' '.join(root.itertext())
                fallback_text = re.sub(r'[ \t\r\f\v]+', ' ', fallback_text) # 规范化
                txt_parts.append(fallback_text)

        elif kind == 'wiley':
            # --- 针对 Wiley 格式的优化处理 ---
            # Wiley 通常基于 JATS，结构和处理逻辑与 JATS 高度相似
            # 区别主要在于参考文献可能使用 <bib> 和 <citation> 结构

            # 1. 查找 <body> 元素 (正文)
            # 定义命名空间映射
            namespaces = {'w': 'http://www.wiley.com/namespaces/wiley'}

            # 1. 查找 <body> 元素 (正文)，使用命名空间前缀 'w:'
            # 注意：根元素是 <component>，所以查找 './/w:body'
            body_elem = root.find('.//w:body', namespaces)

            # 2. 查找 <bibliography> 元素 (在 <body> 内)，同样使用命名空间前缀
            back_elem = None  # 此结构中没有独立的 <back>
            bibliography_elem = None
            ref_list_elem = None  # 也检查是否存在标准 JATS 的 ref-list

            if body_elem is not None:
                # 在 <body> 内查找 <bibliography>，使用命名空间前缀 'w:'
                bibliography_elem = body_elem.find('.//w:bibliography', namespaces)
                # 同时也检查 <body> 内是否有标准的 JATS <ref-list> (以防万一)
                ref_list_elem = body_elem.find('.//w:ref-list', namespaces)
                # 如果标准 JATS 的 ref-list 不带 wiley 命名空间前缀，也可以尝试不带命名空间查找
                if ref_list_elem is None:
                    ref_list_elem = body_elem.find('.//ref-list')

            header_elem = root.find('.//w:header', namespaces)
            if header_elem is None: # Fallback: 不带命名空间查找
                header_elem = root.find('.//header')

            if header_elem is not None:
                header_parts = []
                
                # 1.1 提取标题 (article-title 在 contentMeta/titleGroup/title)
                # 注意：示例中 <w:title> 是命名空间标签
                title_elem = header_elem.find('.//w:contentMeta/w:titleGroup/w:title[@type="main"]', namespaces) or \
                            header_elem.find('.//contentMeta/titleGroup/title[@type="main"]') or \
                            header_elem.find('.//w:contentMeta/w:titleGroup/w:title', namespaces) or \
                            header_elem.find('.//contentMeta/titleGroup/title')
                if title_elem is not None and title_elem.text:
                    title_text = title_elem.text.strip()
                    if title_text:
                        header_parts.append(title_text)
                        header_parts.append('\n\n') # 标题后加双换行

                # 1.2 提取作者 (creators/creator/personName)
                creators_elem = header_elem.find('.//w:contentMeta/w:creators', namespaces) or \
                                header_elem.find('.//contentMeta/creators')
                if creators_elem is not None:
                    # 优先查找 role 为 author 的 creator
                    creator_elems = creators_elem.findall('.//w:creator[@creatorRole="author"]', namespaces) or \
                                    creators_elem.findall('.//creator[@creatorRole="author"]')
                    
                    # 如果没找到带 role 的，则获取所有 creator (除了可能的 editor 等)
                    # 可以通过检查是否有 creatorRole 且不为 author 来过滤，但简化处理，先取所有
                    # 或者更简单地，获取所有 creator，然后在循环里检查 role
                    if not creator_elems:
                         creator_elems = creators_elem.findall('.//w:creator', namespaces) or \
                                         creators_elem.findall('.//creator')
                    
                    author_names = []
                    for creator_elem in creator_elems:
                        # 检查 creatorRole，如果不是 author 则跳过 (如果之前没筛选)
                        role = creator_elem.get(f'{{{namespaces.get("w", "")}}}creatorRole') if namespaces.get("w") else creator_elem.get('creatorRole')
                        # 如果 role 存在但不是 author，则跳过 (可选，根据需求)
                        # if role and role != 'author':
                        #     continue

                        # 直接从 creator 元素拼接姓名
                        # 查找 personName
                        person_name_elem = creator_elem.find('.//w:personName', namespaces) or \
                                           creator_elem.find('.//personName')
                        
                        if person_name_elem is not None:
                            # 提取 givenNames 和 familyName
                            # 注意标签名是 givenNames (驼峰式)，不是 given-names (JATS 风格)
                            given_names_elem = person_name_elem.find('.//w:givenNames', namespaces) or \
                                               person_name_elem.find('.//givenNames')
                            family_name_elem = person_name_elem.find('.//w:familyName', namespaces) or \
                                               person_name_elem.find('.//familyName')
                            
                            given_names_text = given_names_elem.text.strip() if given_names_elem is not None and given_names_elem.text else ""
                            family_name_text = family_name_elem.text.strip() if family_name_elem is not None and family_name_elem.text else ""
                            
                            # 拼接格式：Given Names Family Name
                            if family_name_text or given_names_text: # 至少有一个名字部分
                                full_name = f"{given_names_text} {family_name_text}".strip()
                                author_names.append(full_name)
                        else:
                            # Fallback: 如果 creator 下没有 personName (不太可能)，尝试提取纯文本
                            creator_text = ''.join(creator_elem.itertext()).strip()
                            if creator_text:
                                author_names.append(creator_text)
                                
                    if author_names:
                        # 使用逗号和空格分隔作者，这是常见的格式
                        authors_str = ', '.join(author_names) 
                        header_parts.append(authors_str)
                        header_parts.append('\n')

                # 1.3 提取摘要 (abstractGroup/abstract/p)
                abstract_group_elem = header_elem.find('.//w:contentMeta/w:abstractGroup', namespaces) or \
                                    header_elem.find('.//contentMeta/abstractGroup')
                abstract_elem = None
                if abstract_group_elem is not None:
                    # 可能有多个 abstract，例如不同语言的
                    abstract_elems = abstract_group_elem.findall('.//w:abstract[@type="main"]', namespaces) or \
                                    abstract_group_elem.findall('.//abstract[@type="main"]') or \
                                    abstract_group_elem.findall('.//w:abstract', namespaces) or \
                                    abstract_group_elem.findall('.//abstract')
                    if abstract_elems:
                        abstract_elem = abstract_elems[0] # 取第一个或主要的
                
                # 如果 abstractGroup 下没找到，直接在 contentMeta 下找
                if abstract_elem is None:
                    abstract_elem = header_elem.find('.//w:contentMeta/w:abstract', namespaces) or \
                                    header_elem.find('.//contentMeta/abstract')

                if abstract_elem is not None:
                    # abstract_title_elem = abstract_elem.find('.//w:title', namespaces) or abstract_elem.find('.//title')
                    abstract_title = "Abstract" # 通常标题就是 Abstract，或可以省略
                    # if abstract_title_elem is not None and abstract_title_elem.text:
                    #     abstract_title = abstract_title_elem.text.strip()
                    
                    # 提取摘要内容，通常是 <p> 标签
                    abstract_ps = abstract_elem.findall('.//w:p', namespaces) or abstract_elem.findall('.//p')
                    abstract_texts = []
                    for p_elem in abstract_ps:
                        p_text = ''.join(p_elem.itertext()).strip()
                        if p_text:
                            p_text = re.sub(r'[ \t\r\f\v]+', ' ', p_text) # 规范化段落内空格
                            abstract_texts.append(p_text)
                    if abstract_texts:
                        header_parts.append(abstract_title)
                        header_parts.append('\n')
                        # 用换行连接段落
                        header_parts.append('\n'.join(abstract_texts))
                        header_parts.append('\n\n')

                # --- 将处理好的 <header> 内容添加到主 txt_parts 开头 ---
                if header_parts:
                    txt_parts.extend(header_parts) # 添加 header 内容到 txt_parts
            # --- 处理正文部分 (body) ---
            # 使用与你调试好的 JATS 逻辑完全相同的结构化提取方法
            if body_elem is not None:
                # 定义块级元素标签 (与 JATS 相同)
                direct_block_tags = {'p', 'title', 'list', 'disp-formula', 'fig', 'table-wrap', 'statement', 'boxed-text', 'verse-group'}
                container_tags = {'sec'}

                def process_element(elem: ET.Element, is_within_container: bool = False):
                    """递归处理元素，避免重复提取嵌套块内容 (Wiley/JATS 通用逻辑)"""
                    # 修正点：正确获取不带命名空间的标签名
                    tag_local_name = elem.tag.split('}')[-1] # <--- 修改这里

                    # 1. 检查是否为直接块级元素
                    if tag_local_name in direct_block_tags:
                        block_text = ''.join(elem.itertext()).strip()
                        if block_text:
                            # 规范化块内文本 (合并空格/制表符/换行符为单空格)
                            block_text = re.sub(r'[ \t\r\f\v]+', ' ', block_text)
                            txt_parts.append(block_text)
                            txt_parts.append('\n') # 块后添加换行以保留结构
                        return # 处理完直接块就返回，不处理子元素，避免重复

                    # 2. 检查是否为容器元素 (如 <sec>)
                    elif tag_local_name in container_tags:
                        # 对于容器，我们不直接提取其文本，而是处理其直接子元素
                        for child in elem:
                            process_element(child, is_within_container=True)
                        return # 处理完容器的子元素就返回

                    # 3. 如果不是块也不是容器，但我们在容器内部，则检查其子元素
                    elif is_within_container:
                        # 递归检查子元素
                        for child in elem:
                            process_element(child, is_within_container=True)
                    else:
                        # 顶层非块非容器元素，检查其子元素 (为了健壮性)
                        for child in elem:
                            process_element(child, is_within_container=False)

                # --- 启动处理 <body> ---
                # 遍历 <body> 的直接子元素
                for child_elem in body_elem:
                    process_element(child_elem, is_within_container=False)
            back_parts = []
            # --- 处理参考文献和其他 <back> 内容 ---
            if bibliography_elem is not None:
                # --- 处理 <bibliography> -> <bib> -> <citation> ---
                # 查找 <bib> 元素，使用命名空间前缀
                bib_entries = bibliography_elem.findall('.//w:bib', namespaces)
                # 如果上面找不到，尝试不带命名空间查找
                if not bib_entries:
                    bib_entries = bibliography_elem.findall('.//bib')

                ref_texts = []
                for bib_elem in bib_entries:
                    # 查找 <citation> 元素，使用命名空间前缀
                    citation_elem = bib_elem.find('./w:citation', namespaces)
                    # 如果上面找不到，尝试不带命名空间查找
                    if citation_elem is None:
                        citation_elem = bib_elem.find('./citation')

                    if citation_elem is not None:
                        cit_text = ''.join(citation_elem.itertext()).strip()
                        if cit_text:
                            cit_text = re.sub(r'\s+', ' ', cit_text)
                            ref_texts.append(cit_text)

                if ref_texts:
                    # 检查 <bibliography> 是否有标题
                    has_title = False
                    biblio_title = bibliography_elem.find('./w:title', namespaces) or bibliography_elem.find('./title')
                    if biblio_title is not None and biblio_title.text:
                        title_text = biblio_title.text.strip().lower()
                        if title_text in ('references', 'bibliography', '参考文献'):
                            has_title = True

                    if not has_title:
                        back_parts.append("References")
                        back_parts.append('\n')

                    # 格式化条目
                    formatted_ref_entries = []
                    for entry_text in ref_texts:
                        entry_text = re.sub(r'\b([A-Z])\s+([A-Z][a-z]+)', r'\1. \2', entry_text)
                        entry_text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z][a-z]+)', r'\1. \2. \3', entry_text)
                        if entry_text and entry_text != "\n" and not entry_text.endswith('.'):
                            entry_text += '.'
                        formatted_ref_entries.append(entry_text)
                        formatted_ref_entries.append('\n')

                    back_parts.extend(formatted_ref_entries)

            # --- 提取 <body> 内的其他部分 ---
            # 例如，查找 Acknowledgements
            # 注意：查找路径和元素名都需要考虑命名空间
            ack_elem = body_elem.find('.//w:ack', namespaces) or body_elem.find('.//ack')
            if ack_elem is not None:
                ack_title_elem = ack_elem.find('./w:title', namespaces) or ack_elem.find('./title')
                ack_title = "Acknowledgements"
                if ack_title_elem is not None and ack_title_elem.text:
                    ack_title = ack_title_elem.text.strip()
                ack_content = ''.join(ack_elem.itertext()).strip()
                if ack_title_elem is not None and ack_title_elem.text:
                    ack_title_text = ack_title_elem.text.strip()
                    if ack_content.startswith(ack_title_text):
                        ack_content = ack_content[len(ack_title_text):].strip()
                if ack_content:
                    back_parts.append('\n')
                    back_parts.append(ack_title)
                    back_parts.append('\n')
                    ack_content = re.sub(r'[ \t\r\f\v]+', ' ', ack_content)
                    ack_content = re.sub(r'\s+', ' ', ack_content)
                    back_parts.append(ack_content.strip())
                    back_parts.append('\n')

            if back_parts:
                # txt_parts.append('\n') # 可选：与正文主体分隔
                txt_parts.extend(back_parts)
            # Fallback: 如果没有找到 <body> 也没有参考文献列表，则回退
            if body_elem is None and ref_list_elem is None and bibliography_elem is None:
                print(f"Warning: Could not find <body> or reference list/bibliography in Wiley file {xml_path}. Falling back to full text extraction.")
                fallback_text = ' '.join(root.itertext())
                fallback_text = re.sub(r'[ \t\r\f\v]+', ' ', fallback_text) # 规范化
                txt_parts.append(fallback_text)

        else:
             # 默认：提取所有文本 (不太可能到达这里，因为 kind 总是上述之一)
             txt_parts.append(' '.join(root.itertext()))


        # 4. 合并并清理提取的文本
        txt = ''.join(txt_parts)

        # if txt:
        #     # 规范化换行符 (处理 \r, \r\n)
        #     txt = re.sub(r'\r\n?', '\n', txt)
        #     # 将多个连续的换行符（可能带有空格/制表符）限制为最多两个（段落间空行）
        #     # 但要小心，因为我们自己添加了换行
        #     txt = re.sub(r'[ \t]*\n([ \t]*\n)+', '\n\n', txt) # 将多个换行（带空格）合并为两个
        #     # txt = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', txt) # 限制最多两个连续空行
        #     # 规范化段落内的空格（不包括换行符）
        #     # txt = re.sub(r'[ \t]+', ' ', txt) # 这已经在提取时做过了，可以省略或微调
        #     # 移除行首行尾的空格 (但保留空行)
        #     lines = txt.split('\n')
        #     cleaned_lines = [line.rstrip() for line in lines] # 只移除行尾空格，保留行首空格可能的缩进意义？
        #                           # 或者 .strip() 移除首尾所有空格
        #     # cleaned_lines = [line.strip() for line in lines]
        #     txt = '\n'.join(cleaned_lines)

        #     # 应用通用的文本清理和链接修复
        #     txt = text_clean(txt)
        #     txt = fix_broken_links(txt)
        #     # 注意：extract_text 函数内部也会调用 format_references
        #     # 如果在这里调用，要确保不会重复处理或冲突
        #     # txt = format_references(txt)

        return txt.strip()

    except Exception as e:
        print(f"Error in xml2text_optimized for {xml_path}: {e}")
        return ""
def _extract_biblstruct_text(bibl_elem: ET.Element) -> str:
    """
    从 TEI <biblstruct> 元素中提取文本，并将其合并为单行。
    """
    if bibl_elem is None:
        return ""
    try:
        # 1. 提取 <biblstruct> 内部的所有文本
        bibl_text = ''.join(bibl_elem.itertext()).strip()

        # 2. 规范化空白字符：将所有连续的空白（空格、制表符、换行符）合并为单个空格
        # 这确保了整个条目在文本中是单行的
        if bibl_text:
            bibl_text = re.sub(r'\s+', ' ', bibl_text)

        # 3. （可选）进行一些基本的清理，例如移除多余的点或空格
        # bibl_text = re.sub(r'\s*\.\s*', '. ', bibl_text) # 规范化句号周围的空格
        # bibl_text = re.sub(r'\s+,', ',', bibl_text) # 移除逗号前的空格
        # bibl_text = bibl_text.strip()

        return bibl_text
    except Exception as e:
        print(f"Error extracting text from <biblstruct>: {e}")
        # 如果出错，回退到简单提取
        try:
            return ' '.join(bibl_elem.itertext()).strip()
        except:
            return ""
if __name__ == "__main__":
    xml_path = "10.1111_mec.16977.xml"
    
    # 使用修改后的函数
    full_text = xml2text(xml_path)
    
    print(f"Length of extracted text: {len(full_text)}")
    print("--- Extracted Text ---")
    print(full_text)
