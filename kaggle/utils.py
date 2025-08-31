import re

def smart_chunker(text: str, min_size: int, max_size: int, overlap: int) -> list[str]:
    
    # 1. 使用Markdown标题分割文本
    raw_sections = re.split(r'(^#{1,6} .*)', text, flags=re.MULTILINE)
    
    # 2. 构建逻辑块
    logical_blocks = []
    current_block = ""
    for section in raw_sections:
        section_clean = section.strip()
        if not section_clean:
            continue
        if section_clean.startswith('#'):
            if current_block:
                logical_blocks.append(current_block)
            current_block = ""  # 开始新块，但不包含标题
        else:
            current_block += section
    if current_block:
        logical_blocks.append(current_block.strip())

    # 3. 优化的小块合并逻辑
    merged_blocks = []
    i = 0
    n = len(logical_blocks)
    
    while i < n:
        if len(logical_blocks[i]) >= max_size:
            merged_blocks.append(logical_blocks[i])
            i += 1
            continue
        
        merged_str = logical_blocks[i]
        last_valid = None
        j = i + 1
        
        while j < n:
            next_block = logical_blocks[j]
            candidate = f"{merged_str}\n\n{next_block}"
            
            if len(candidate) > max_size:
                break
                
            merged_str = candidate
            if len(merged_str) >= min_size:
                last_valid = merged_str
            j += 1
        
        if j > i + 1:
            merged_blocks.append(last_valid if last_valid else merged_str)
            i = j
        else:
            merged_blocks.append(logical_blocks[i])
            i += 1

    # 4. 增强的边界处理（头部+尾部）
    final_chunks = []
    for block in merged_blocks:
        if len(block) <= max_size:
            final_chunks.append(block)
            continue
            
        current_pos = 0
        while current_pos < len(block):
            start_pos = current_pos
            end_pos = min(current_pos + max_size, len(block))
            
            # 处理剩余文本
            if len(block) - current_pos <= max_size:
                final_chunks.append(block[current_pos:])
                break
                
            # ===== 关键改进：头部边界检测 =====
            # 只有在非第一个块时才需要进行头部边界调整
            if current_pos > 0:
                head_search_end = min(start_pos + overlap, len(block))
                head_adjustment = 0
                
                # 跳过头部残缺句子（但不跳过正常的空格和换行）
                cn_head_delimiters = ['。', '！', '？', '；', '……', '，', '.', '!', '?', ';', ',']
                for pos in range(start_pos, head_search_end):
                    if block[pos] in cn_head_delimiters:
                        head_adjustment = pos + 1
                        break
                
                # 如果没找到标点，寻找单词边界（避免切断单词）
                if head_adjustment == 0:
                    # 向前查找空格，确保不切断单词
                    for pos in range(start_pos, min(start_pos + 20, len(block))):
                        if block[pos] == ' ':
                            head_adjustment = pos + 1
                            break
                            
                if head_adjustment > 0:
                    start_pos = head_adjustment
                    end_pos = min(start_pos + max_size, len(block))
            
            # ===== 尾部边界检测 =====
            search_start = max(start_pos, end_pos - overlap)
            best_split_pos = -1
            
            # 优先检测：换行符（段落边界）
            nl_pos = block.rfind('\n', search_start, end_pos)
            if nl_pos > best_split_pos:
                best_split_pos = nl_pos
                
            # 其次检测：中文句子边界
            cn_delimiters = ['。', '！', '？', '；', '……','，']
            for delim in cn_delimiters:
                pos = block.rfind(delim, search_start, end_pos)
                if pos > best_split_pos:
                    best_split_pos = pos
                    
            # 英文句子边界检测
            en_delimiters = ['.', '!', '?', ';', ',']
            for delim in en_delimiters:
                pos = block.rfind(delim, search_start, end_pos)
                if pos > best_split_pos:
                    # 验证是否真实句子结尾
                    if (pos == len(block)-1 or 
                        block[pos+1] in [' ', '\n', ')', ']', '>', '"', "'"] or
                        (pos+2 < len(block) and block[pos+1].isupper())):
                        best_split_pos = pos
            
            # 确定分割点
            if best_split_pos > start_pos:
                split_pos = best_split_pos + 1
                chunk_content = block[start_pos:split_pos].strip()
                if chunk_content:  # 避免空块
                    final_chunks.append(chunk_content)
                # 改进：确保下一个块从合适的边界开始
                next_pos = max(split_pos - overlap, current_pos)
                # 智能边界检测，避免分割URL、邮箱等
                while next_pos < len(block) and next_pos < split_pos:
                    # 扩大搜索范围检查是否在URL或链接中间
                    context_before = block[max(0, next_pos-30):next_pos]
                    context_after = block[next_pos:min(len(block), next_pos+30)]
                    full_context = context_before + context_after
                    
                    # 检查是否在URL、DOI、链接结构中间
                    if ('://' in full_context or 
                        'doi.org' in full_context or 
                        'https' in full_context or
                        '[' in context_before and ']' in context_after or
                        '(' in context_before and ')' in context_after):
                        # 如果在复杂结构中，寻找真正的单词边界
                        while next_pos < len(block) and block[next_pos] not in [' ', '\n', '.', '!', '?'] or (next_pos < len(block)-1 and block[next_pos] == '.' and not block[next_pos+1].isspace()):
                            next_pos += 1
                        # 找到空格或句子结尾
                        while next_pos < len(block) and block[next_pos] in [' ', '\t']:
                            next_pos += 1
                        break
                    elif block[next_pos] in [' ', '\n']:
                        break
                    next_pos += 1
                current_pos = next_pos
            else:  # 无合适边界
                chunk_content = block[start_pos:end_pos].strip()
                if chunk_content:  # 避免空块
                    final_chunks.append(chunk_content)
                # 改进：确保下一个块从合适的边界开始  
                next_pos = max(end_pos - overlap, current_pos)
                # 智能边界检测，避免分割URL、邮箱等
                while next_pos < len(block) and next_pos < end_pos:
                    # 扩大搜索范围检查是否在URL或链接中间
                    context_before = block[max(0, next_pos-30):next_pos]
                    context_after = block[next_pos:min(len(block), next_pos+30)]
                    full_context = context_before + context_after
                    
                    # 检查是否在URL、DOI、链接结构中间
                    if ('://' in full_context or 
                        'doi.org' in full_context or 
                        'https' in full_context or
                        '[' in context_before and ']' in context_after or
                        '(' in context_before and ')' in context_after):
                        # 如果在复杂结构中，寻找真正的单词边界
                        while next_pos < len(block) and block[next_pos] not in [' ', '\n', '.', '!', '?'] or (next_pos < len(block)-1 and block[next_pos] == '.' and not block[next_pos+1].isspace()):
                            next_pos += 1
                        # 找到空格或句子结尾
                        while next_pos < len(block) and block[next_pos] in [' ', '\t']:
                            next_pos += 1
                        break
                    elif block[next_pos] in [' ', '\n']:
                        break
                    next_pos += 1
                current_pos = next_pos
                
    return final_chunks
def remove_references_section(text):
    lines = text.split('\n')
    cut_index = -1
    
    # Look backwards from end of document
    for i in range(len(lines) - 1, max(0, int(len(lines) * 0.3)), -1):
        line = lines[i].strip()
        
        obvious_patterns = [
            # References patterns
            r'^REFERENCES?$',
            r'^\d+\.?\s+REFERENCES?$',
            r'^\d+\.?\s+References?$',
            r'^References?:?$',
            
            # Bibliography patterns  
            r'^BIBLIOGRAPHY$',
            r'^\d+\.?\s+BIBLIOGRAPHY$',
            r'^\d+\.?\s+Bibliography$',
            r'^Bibliography:?$',
            
            # Other common patterns
            r'^Literature\s+Cited$',
            r'^Works\s+Cited$',
            r'^ACKNOWLEDGMENTS?$',
            r'^Acknowledgments?$',
            r'^FUNDING$',
            r'^CONFLICTS?\s+OF\s+INTEREST$',
        ]
        
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in obvious_patterns):
            # Double-check: look at following lines for citation patterns
            following_lines = lines[i+1:i+5]  # Check more lines
            has_citations = False
            
            for follow_line in following_lines:
                if follow_line.strip():
                    # Check for obvious citation patterns
                    if (re.search(r'\(\d{4}\)', follow_line) or
                        re.search(r'\d{4}\.', follow_line) or
                        'doi:' in follow_line.lower() or
                        ' et al' in follow_line.lower() or
                        re.search(r'^\[\d+\]', follow_line.strip()) or  # [1], [2], etc.
                        re.search(r'^\d+\.', follow_line.strip())):     # 1., 2., etc.
                        has_citations = True
                        break
            
            # Only cut if we found citation-like content
            if has_citations or i >= len(lines) - 3:  # Or very near end
                cut_index = i
                break
    
    if cut_index != -1:
        return '\n'.join(lines[:cut_index]).strip()
    
    return text.strip()
def remove_references_section_markdown(text: str) -> str:
    """
    移除标准Markdown格式文档中的参考文献部分
    
    功能说明：
    - 专门针对标准Markdown格式设计
    - 识别参考文献标题（支持各种Markdown标题格式和粗体）
    - 删除从参考文献标题开始到下一个同级或更高级标题之前的所有内容
    - 如果没有下一个标题，则删除到文档末尾
    
    参数:
        text (str): Markdown格式的文档文本
        
    返回:
        str: 移除参考文献部分后的文本
        
    示例:
        输入包含:
        ## Introduction
        content...
        
        ## **References**
        - ref1
        - ref2
        
        ## Appendix
        content...
        
        输出：删除References部分，保留Introduction和Appendix
    """
    lines = text.split('\n')
    ref_start_index = -1
    ref_level = 0
    
    # 参考文献标题的正则表达式模式
    reference_patterns = [
        # 标准Markdown标题格式
        r'^(#{1,6})\s*References?$',                    # # References, ## References
        r'^(#{1,6})\s*REFERENCES?$',                   # # REFERENCES, ## REFERENCES
        r'^(#{1,6})\s*Bibliography$',                  # # Bibliography
        r'^(#{1,6})\s*BIBLIOGRAPHY$',                  # # BIBLIOGRAPHY
        
        # 带粗体的Markdown标题格式
        r'^(#{1,6})\s*\*\*References?\*\*$',          # # **References**
        r'^(#{1,6})\s*\*\*REFERENCES?\*\*$',         # # **REFERENCES**
        r'^(#{1,6})\s*\*\*Bibliography\*\*$',        # # **Bibliography**
        r'^(#{1,6})\s*\*\*BIBLIOGRAPHY\*\*$',        # # **BIBLIOGRAPHY**
        
        # 其他常见格式
        r'^(#{1,6})\s*Literature\s+Cited$',          # # Literature Cited
        r'^(#{1,6})\s*Works\s+Cited$',               # # Works Cited
    ]
    
    # 第一步：找到参考文献标题
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        for pattern in reference_patterns:
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                ref_start_index = i
                ref_level = len(match.group(1))  # 获取标题级别（#的数量）
                break
        
        if ref_start_index != -1:
            break
    
    if ref_start_index == -1:
        return text  # 没有找到参考文献标题
    
    # 第二步：找到下一个同级或更高级的标题
    end_index = len(lines)  # 默认删除到文档末尾
    
    for i in range(ref_start_index + 1, len(lines)):
        line_stripped = lines[i].strip()
        
        # 检查是否是Markdown标题
        title_match = re.match(r'^(#{1,6})\s+', line_stripped)
        if title_match:
            title_level = len(title_match.group(1))
            # 如果是同级或更高级标题，则在此处结束删除
            if title_level <= ref_level:
                end_index = i
                break
    
    # 第三步：删除参考文献部分
    if ref_start_index < end_index:
        # 保留参考文献标题之前的内容和下一个标题之后的内容
        result_lines = lines[:ref_start_index]
        # result_lines = lines[:ref_start_index] + lines[end_index:]
        return '\n'.join(result_lines).strip()
    
    return text.strip()
def clean_json_output(raw_text: str) -> str:
    """移除 Markdown 代码块标记，提取纯 JSON"""
    # 尝试匹配 ```json ... ``` 之间的内容
    match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 返回纯 JSON 部分
    return raw_text  # 如果没有 Markdown 标记，返回原文本
def remove_brackets_before_urls(text: str) -> str:
    """
    移除URL前面紧挨着的方括号内容
    
    功能说明：
    - 识别 http:// 或 https:// 开头的URL
    - 如果URL前面紧挨着 [...] 格式的内容，将方括号部分删除
    - 保留圆括号和URL本身
    
    参数:
        text (str): 需要处理的文本
        
    返回:
        str: 处理后的文本
        
    示例:
        输入: "查看 [点击这里](http://example.com) 获取更多信息"
        输出: "查看 (http://example.com) 获取更多信息"
        
        输入: "[GitHub链接](https://github.com/user/repo) 和 [官网](http://website.com)"
        输出: "(https://github.com/user/repo) 和 (http://website.com)"
    """
    # 匹配 [任意内容](http://... 或 https://...) 的模式
    # \[.*?\] 匹配方括号及其内容（非贪婪）
    # \((https?://[^\s\)]+)\) 匹配圆括号内的URL并捕获
    pattern = r'\[([^\]]{1,30})\]\((https?://[^\s\)]+)\)' # \[([^\]]*)\]\(([^\s\)]+)\)
    
    # 使用替换，保留圆括号和URL部分
    result = re.sub(pattern, r'(\2)', text)
    
    # 检查并删除重复的URL（即使中间有空格或其他字符）
    # 使用循环来处理多个重复的情况
    
    # 先处理直接相邻的情况：(url)(url)
    duplicate_pattern = r'\((https?://[^\s\)]+)\)\(\1\)'
    while re.search(duplicate_pattern, result):
        # result = re.sub(duplicate_pattern, r'(\1)', result)
        result = re.sub(duplicate_pattern, r'', result)
    
    # 再处理中间有空格或少量字符的情况：(url) 任意字符 (url)
    # 限制中间的字符不超过10个，避免误删除距离很远的URL
    spaced_duplicate_pattern = r'\((https?://[^\s\)]+)\)([^(]{0,10}?)\(\1\)'
    while re.search(spaced_duplicate_pattern, result):
        result = re.sub(spaced_duplicate_pattern, r'(\1)\2', result)
    
    return result
def replace_page_references_with_urls(text: str) -> str:
    """
    将文本中的(#page)引用替换为参考文献中对应的DOI
    
    功能说明：
    1. 识别文本中的 (#page-x-y) 格式引用
    2. 在参考文献部分找到对应的 <span id="page-x-y"></span> 条目
    3. 从span标签开始往后搜索所有DOI，直到遇到边界：
       - 下一个 <span id= 标签
       - 两个连续换行符（段落分隔）
       - # 标题
    4. 将找到的多个DOI用英文逗号连接，替换 (#page-x-y)
    5. 如果没有找到DOI则删除引用
    
    参数:
        text (str): 包含页面引用和参考文献的完整文本
        
    返回:
        str: 替换后的文本
        
    示例:
        单个DOI的情况:
        输入: [2](#page-5-8) 和对应的参考文献条目
        输出: [2](10.7937/TCIA.e3sv-re93)
        
        多个DOI的情况:
        输入: [2](#page-5-8) 对应多个相关条目
        输出: [2](10.1000/example1,10.1038/s41416-018-0185-8)
        
        无DOI的情况:
        输入: [1](#page-5-0) 但没有对应的DOI
        输出: [1] (引用被删除)
    """
    
    # 第一步：提取所有的页面引用
    page_ref_pattern = r'\(#page-[\w-]+\)'
    page_references = re.findall(page_ref_pattern, text)
    
    if not page_references:
        return text
    
    # 第二步：为每个页面引用找到对应的URL
    result_text = text
    
    for page_ref in page_references:
        # 提取页面ID (去掉括号和#)
        page_id = page_ref[2:-1]  # 从 "(#page-5-8)" 提取 "page-5-8"
        
        # 查找对应的span标签和其后的URL
        span_pattern = rf'<span id="{re.escape(page_id)}"></span>'
        span_match = re.search(span_pattern, text)
        
        if span_match:
            # 从span标签位置开始，往后查找到下一个边界
            text_after_span = text[span_match.end():]
            
            # 定义搜索边界：下一个<span id=、两个连续换行符、或# 标题
            boundary_patterns = [
                r'<span id=',              # 下一个span标签
                r'\n\n',                   # 两个换行符（段落分隔）
                r'\n#+\s',                 # # 标题
            ]
            
            # 找到最近的边界位置
            earliest_boundary = len(text_after_span)  # 默认搜索到文本末尾
            
            for pattern in boundary_patterns:
                match = re.search(pattern, text_after_span)
                if match and match.start() < earliest_boundary:
                    earliest_boundary = match.start()
            
            # 在边界内搜索URL
            search_text = text_after_span[:earliest_boundary]
            
            # 只匹配DOI格式
            url_patterns = [
                (r'(10\.\d{4,}/(?:[-._;/:A-Z0-9]*(?:\([-._;/:A-Z0-9]*\)[-._;/:A-Z0-9]*)*[A-Z0-9]|[-._;/:A-Z0-9]*[A-Z0-9]))', 0, re.IGNORECASE),  # DOI格式
            ]
            
            found_urls = []
            
            # 查找所有DOI
            for pattern_info in url_patterns:
                if len(pattern_info) == 3:
                    url_pattern, group_index, flags = pattern_info
                    url_matches = re.finditer(url_pattern, search_text, flags)
                else:
                    url_pattern, group_index = pattern_info
                    url_matches = re.finditer(url_pattern, search_text)
                
                for url_match in url_matches:
                    if group_index == 0:
                        url = url_match.group(0)  # 使用整个匹配
                    else:
                        url = url_match.group(group_index)  # 使用指定组
                    
                    # 清理URL，去掉可能的尾部标点
                    url = url.rstrip('.,;:')
                    
                    # 避免重复添加相同的URL
                    if url and url not in found_urls:
                        found_urls.append(url)
            
            # 将找到的所有URL组合成一个字符串
            if found_urls:
                if len(found_urls) == 1:
                    found_url = found_urls[0]
                else:
                    found_url = ','.join(found_urls)
            else:
                found_url = None
        else:
            print(page_ref,"不行")
            found_url = None
        # 进行替换：有URL就替换为URL，没有URL就替换为空字符串
        if found_url:
            result_text = result_text.replace(page_ref, f'({found_url.lower()})')
        else:
            result_text = result_text.replace(page_ref, '')
    
    return result_text
def remove_brackets_before_dois(text: str) -> str:
    """
    移除DOI前面紧挨着的方括号内容
    
    功能说明：
    - 识别 10.xxxx/... 格式的DOI
    - 如果DOI前面紧挨着 [...] 格式的内容，将方括号部分删除
    - 保留圆括号和DOI本身
    
    参数:
        text (str): 需要处理的文本
        
    返回:
        str: 处理后的文本
        
    示例:
        输入: "参考文献 [引用1](10.1038/s41416-018-0185-8) 和相关研究"
        输出: "参考文献 (10.1038/s41416-018-0185-8) 和相关研究"
        
        输入: "[DOI链接](10.7937/TCIA.e3sv-re93) 和 [另一个](10.1000/example)"
        输出: "(10.7937/TCIA.e3sv-re93) 和 (10.1000/example)"
    """
    # 匹配 [任意内容](10.xxxx/...) 的DOI模式
    # 使用完整的DOI正则表达式模式，添加逗号支持
    doi_pattern = r'\[([^\]]{1,30})\]\((10\.\d{4,}/(?:[-._;/:,A-Z0-9]*(?:\([-._;/:,A-Z0-9]*\)[-._;/:,A-Z0-9]*)*[A-Z0-9]|[-._;/:,A-Z0-9]*[A-Z0-9]))\)'
    
    # 使用替换，保留圆括号和DOI部分
    result = re.sub(doi_pattern, r'(\2)', text, flags=re.IGNORECASE)
    
    # 检查并删除重复的DOI（即使中间有空格或其他字符）
    # 使用循环来处理多个重复的情况
    
    # 先处理直接相邻的情况：(doi)(doi)
    duplicate_pattern = r'\((10\.\d{4,}/[^\)]+)\)\(\1\)'
    while re.search(duplicate_pattern, result, re.IGNORECASE):
        result = re.sub(duplicate_pattern, r'(\1)', result, flags=re.IGNORECASE)
    
    # 再处理中间有空格或少量字符的情况：(doi) 任意字符 (doi)
    # 限制中间的字符不超过10个，避免误删除距离很远的DOI
    spaced_duplicate_pattern = r'\((10\.\d{4,}/[^\)]+)\)([^(]{0,10}?)\(\1\)'
    while re.search(spaced_duplicate_pattern, result, re.IGNORECASE):
        result = re.sub(spaced_duplicate_pattern, r'(\1)\2', result, flags=re.IGNORECASE)
    
    return result
def remove_citations_by_year_pattern(text: str) -> str:
    """
    第二步：基于年份格式删除参考文献条目
    
    功能：
    - 识别包含 (年份). 或 (年份), 格式的行
    - 删除这些行，不管它们在文档的哪个位置
    - 不依赖标题，纯粹基于内容格式判断
    
    参数:
        text (str): 文档文本
        
    返回:
        str: 删除包含年份格式的行后的文本
    """
    lines = text.split('\n')
    filtered_lines = []
    
    # 年份格式的正则表达式：(YYYY). 或 (YYYY), 或 (YYYY)
    year_pattern = r'\(\d{4}[ab]?\)\.'
    
    for line in lines:
        # 如果行中包含年份格式，跳过这一行
        if re.search(year_pattern, line):
            continue
        # 否则保留这一行
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines).strip()
def remove_span_citations(text: str) -> str:
    """
    第三步：删除带有span标签的参考文献条目
    
    功能：
    - 识别以 "- <span id="page-x-y"></span>" 开头的参考文献行
    - 删除这些完整的参考文献条目
    - 支持多行参考文献（如果引用跨越多行）
    
    参数:
        text (str): 文档文本
        
    返回:
        str: 删除span格式参考文献后的文本
    """
    lines = text.split('\n')
    filtered_lines = []
    i = 0
    
    # span参考文献格式的正则表达式
    span_citation_pattern = r'^-\s*<span\s+id="page-[\w-]+"></span>'
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 检查是否是span格式的参考文献开头
        if re.match(span_citation_pattern, line):
            # 跳过这一行（span格式的参考文献）
            i += 1
            continue
        
        # 保留非span格式的行
        filtered_lines.append(lines[i])
        i += 1
    
    return '\n'.join(filtered_lines).strip()
def find_context_boundaries(text: str, match_start: int, match_end: int, target_before: int, target_after: int) -> tuple[int, int]:
    """
    智能查找匹配项的上下文边界，优先确保上下文达到目标长度，然后寻找最远离匹配项的分割点以保留完整语义。
    
    Args:
        text (str): 完整文本。
        match_start (int): 匹配项的起始位置。
        match_end (int): 匹配项的结束位置。
        target_before (int): 期望在匹配项前包含的字符数。
        target_after (int): 期望在匹配项后包含的字符数。

    Returns:
        tuple[int, int]: 上下文的实际起始和结束位置。
    """
    # 定义句子和段落的边界符
    delimiters = ['.', '!', '?', '\n', '。', '！', '？']
    
    # 1. 定义足够大的"草稿"窗口
    draft_start = max(0, match_start - target_before * 2)  # 扩大搜索范围
    draft_end = min(len(text), match_end + target_after * 2)
    
    # 2. 寻找最远离匹配项的分割点
    # --- 确定最终的起始位置 ---
    final_start = draft_start
    best_start_pos = -1  # 记录最远离匹配项的边界位置
    
    for delim in delimiters:
        # 从草稿窗口开始向前查找第一个边界符(最远离匹配项)
        pos = text.find(delim, draft_start, match_start)
        if pos != -1 and (best_start_pos == -1 or pos < best_start_pos):
            best_start_pos = pos
    
    if best_start_pos != -1:
        # 选择最远的边界(最小的位置)，并包含边界后的内容
        final_start = best_start_pos + 1
    
    # 跳过起始位置后的空白字符
    while final_start < len(text) and final_start < match_start and text[final_start].isspace():
        final_start += 1

    # --- 确定最终的结束位置 ---
    final_end = draft_end
    best_end_pos = -1  # 记录最远离匹配项的边界位置
    
    for delim in delimiters:
        # 从匹配项结束向后查找最后一个边界符(最远离匹配项)
        pos = text.rfind(delim, match_end, draft_end)
        if pos > best_end_pos:
            best_end_pos = pos
    
    if best_end_pos != -1:
        # 选择最远的边界(最大的位置)，并包含边界符本身
        final_end = best_end_pos + 1
    
    # 3. 如果找不到边界，回退到原始窗口
    if best_start_pos == -1:
        final_start = max(0, match_start - target_before)
    if best_end_pos == -1:
        final_end = min(len(text), match_end + target_after)
    
    # 确保边界有效
    final_start = max(0, final_start)
    final_end = min(len(text), final_end)
    
    return final_start, final_end