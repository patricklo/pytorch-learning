import re

def smart_chunker(text: str, min_size: int, max_size: int, overlap: int) -> list[str]:
    # 1. 使用Markdown标题分割文本
    raw_sections = re.split(r'(\n^#{1,6} .*)', text, flags=re.MULTILINE)
    
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
            current_block = section
        else:
            current_block += section
    if current_block:
        logical_blocks.append(current_block.strip())

    # 3. 合并小块达到最小尺寸
    merged_blocks = []
    temp_buffer = ""
    for block in logical_blocks:
        if not temp_buffer:
            temp_buffer = block
        else:
            temp_buffer += "\n\n" + block
        
        if len(temp_buffer) >= min_size:
            merged_blocks.append(temp_buffer)
            temp_buffer = ""
            
    if temp_buffer:
        if merged_blocks:
            merged_blocks[-1] += "\n\n" + temp_buffer
        else:
            merged_blocks.append(temp_buffer)

    final_chunks = []
    # 定义句子结束符（包括中英文）
    sentence_delimiters = ['。', '！', '？', '\n']  # 英文标点单独处理

    # 4. 分割大块为符合最大尺寸的块
    for block in merged_blocks:
        if len(block) <= max_size:
            final_chunks.append(block)
            continue
            
        current_pos = 0
        while current_pos < len(block):
            end_pos = min(current_pos + max_size, len(block))
            
            # 如果剩余部分可以直接作为一个块
            if len(block) - current_pos <= max_size:
                final_chunks.append(block[current_pos:])
                break
                
            # 寻找最佳分割点
            best_split_pos = -1
            
            # 首先：处理非英文标点（中文标点和换行符）
            non_en_delimiters = [d for d in sentence_delimiters]
            for delim in non_en_delimiters:
                # 在重叠区域内搜索分隔符
                search_start = max(current_pos, end_pos - overlap)
                pos = block.rfind(delim, search_start, end_pos)
                if pos > best_split_pos:
                    best_split_pos = pos
                    
            # 其次：处理英文结束标点（需要上下文检查）
            search_start = max(current_pos, end_pos - overlap)
            # 在重叠区域内从后向前搜索
            pos = end_pos - 1
            found_en = -1
            while pos >= search_start:
                if block[pos] in ['.', '!', '?']:
                    # 检查上下文：句点后是空格/换行/文本结束
                    if pos == len(block) - 1:  # 文本结束
                        found_en = pos
                        break
                    next_char = block[pos + 1]
                    if next_char in [' ', '\n']:  # 后跟空格或换行
                        found_en = pos
                        break
                pos -= 1
                
            if found_en > best_split_pos:
                best_split_pos = found_en
                
            # 确定分割位置
            if best_split_pos != -1:
                # 分割点后移以包含结束标点
                split_point = best_split_pos + 1
                chunk = block[current_pos:split_point]
                final_chunks.append(chunk.strip())
                # 更新位置（考虑重叠）
                next_start = max(current_pos, split_point - overlap)
                current_pos = next_start
            else:
                # 没有找到合适分割点，硬分割
                chunk = block[current_pos:end_pos]
                final_chunks.append(chunk.strip())
                current_pos = end_pos - overlap  # 应用重叠
                
            # 防止无限循环
            if current_pos >= len(block) - 1:
                break
                
    return final_chunks
def remove_references_section(text):
    lines = text.split('\n')
    cut_index = -1
    
    # Look backwards from end of document
    for i in range(len(lines) - 1, max(0, int(len(lines) * 0.3)), -1):
        line = lines[i].strip()
        
        obvious_patterns = [
            # References patterns
            r'^REFERENCES?$',                    # All caps, alone
            r'^\d+\.?\s+REFERENCES?$',          # Numbered, all caps
            r'^\d+\.?\s+References?$',          # Numbered, title case
            r'^References?:$',                   # With colon
            
            # Bibliography patterns
            r'^BIBLIOGRAPHY$',                   # All caps, alone
            r'^\d+\.?\s+BIBLIOGRAPHY$',         # Numbered, all caps
            r'^\d+\.?\s+Bibliography$',         # Numbered, title case
            r'^Bibliography:$',                  # With colon
            
            # Other common patterns
            r'^Literature\s+Cited$',            # Literature Cited
            r'^Works\s+Cited$',                 # Works Cited
        ]
        
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in obvious_patterns):
            # Double-check: look at following lines for citation patterns
            following_lines = lines[i+1:i+4]
            has_citations = False
            
            for follow_line in following_lines:
                if follow_line.strip():
                    # Check for obvious citation patterns
                    if (re.search(r'\(\d{4}\)', follow_line) or    # (2020)
                        re.search(r'\d{4}\.', follow_line) or       # 2020.
                        'doi:' in follow_line.lower() or           # DOI
                        ' et al' in follow_line.lower()):          # et al
                        has_citations = True
                        break
            
            # Only cut if we found citation-like content
            if has_citations or i >= len(lines) - 3:  # Or very near end
                cut_index = i
                break
    
    if cut_index != -1:
        return '\n'.join(lines[:cut_index]).strip()
    
    return text.strip()
def clean_json_output(raw_text: str) -> str:
    """移除 Markdown 代码块标记，提取纯 JSON"""
    # 尝试匹配 ```json ... ``` 之间的内容
    match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 返回纯 JSON 部分
    return raw_text  # 如果没有 Markdown 标记，返回原文本