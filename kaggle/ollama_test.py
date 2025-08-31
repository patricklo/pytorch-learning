import ollama

# 確保 Ollama 伺服器已在本地運行
# 如果沒有，請在終端機中執行：ollama serve

response = ollama.generate(model='phi3', prompt='幫我寫一首關於海洋的五言絕句。')

print(response['response'])