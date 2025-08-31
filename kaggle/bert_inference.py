import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import re
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

# --- FastText分类头类定义 (从训练脚本复制) ---
class FastTextClassificationHead(nn.Module):
    """
    FastText风格的分类头
    特点：
    1. 使用平均池化而不是CLS token
    2. 简单的线性层结构
    3. 可选的dropout和normalization
    """
    def __init__(self, config, classifier_dropout=0.1):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(classifier_dropout)
        
        # fastText风格：简单的线性分类层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 可选的层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        # fastText风格：使用平均池化而不是CLS token
        if attention_mask is not None:
            # 考虑attention mask的平均池化
            attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            # 简单平均池化
            pooled_output = torch.mean(hidden_states, dim=1)
        
        # 层归一化
        pooled_output = self.layer_norm(pooled_output)
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return logits


class TextCNNClassificationHead(nn.Module):
    """
    TextCNN风格的分类头
    特点：
    1. 使用多个不同大小的卷积核捕获不同尺度的局部特征
    2. 每个卷积核后跟最大池化
    3. 将所有特征拼接后通过全连接层分类
    4. 对短文本分类效果很好
    """
    def __init__(self, config, filter_sizes=[3, 4, 5], num_filters=128, classifier_dropout=0.1):
        super().__init__()
        self.config = config
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        # 多个卷积层，每个卷积核大小不同
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=config.hidden_size, 
                     out_channels=num_filters, 
                     kernel_size=filter_size)
            for filter_size in filter_sizes
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 全连接分类层
        # 输入维度是所有卷积核输出的拼接
        total_num_filters = num_filters * len(filter_sizes)
        self.classifier = nn.Linear(total_num_filters, config.num_labels)
        
        # 可选的批归一化
        self.batch_norm = nn.BatchNorm1d(total_num_filters)
        
    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # 如果有attention_mask，将padding位置设为0
        if attention_mask is not None:
            # 扩展attention_mask到hidden_size维度
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states * attention_mask_expanded
        
        # 转置为卷积所需的格式: (batch_size, hidden_size, seq_len)
        x = hidden_states.transpose(1, 2)
        
        # 存储每个卷积核的输出
        conv_outputs = []
        
        for conv in self.convs:
            # 卷积操作: (batch_size, num_filters, seq_len - filter_size + 1)
            conv_out = F.relu(conv(x))
            
            # 最大池化: (batch_size, num_filters, 1)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            
            # 压缩最后一维: (batch_size, num_filters)
            pooled = pooled.squeeze(2)
            
            conv_outputs.append(pooled)
        
        # 拼接所有卷积核的输出: (batch_size, total_num_filters)
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # 批归一化
        concatenated = self.batch_norm(concatenated)
        
        # Dropout
        concatenated = self.dropout(concatenated)
        
        # 分类
        logits = self.classifier(concatenated)
        
        return logits


class SciBertWithFastTextHead(nn.Module):
    """
    结合BERT编码器和FastText风格分类头的模型
    """
    def __init__(self, model_name, num_labels, classifier_dropout=0.1):
        super().__init__()
        
        # 加载预训练的BERT模型（不包含分类头）
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.config.num_labels = num_labels
        
        # 使用fastText风格的分类头
        self.classifier = FastTextClassificationHead(self.config, classifier_dropout)
        
        # 初始化权重
        self.classifier.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.classifier.bias.data.zero_()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取序列输出（所有token的hidden states）
        sequence_output = outputs.last_hidden_state
        
        # 使用fastText风格的分类头
        logits = self.classifier(sequence_output, attention_mask)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
        )


class SciBertWithTextCNNHead(nn.Module):
    """
    结合BERT编码器和TextCNN风格分类头的模型
    """
    def __init__(self, model_name, num_labels, filter_sizes=[3, 4, 5], num_filters=128, classifier_dropout=0.1):
        super().__init__()
        
        # 加载预训练的BERT模型（不包含分类头）
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.config.num_labels = num_labels
        
        # 使用TextCNN风格的分类头
        self.classifier = TextCNNClassificationHead(
            self.config, 
            filter_sizes=filter_sizes, 
            num_filters=num_filters,
            classifier_dropout=classifier_dropout
        )
        
        # 初始化权重
        for conv in self.classifier.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        nn.init.xavier_uniform_(self.classifier.classifier.weight)
        nn.init.zeros_(self.classifier.classifier.bias)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取序列输出（所有token的hidden states）
        sequence_output = outputs.last_hidden_state
        
        # 使用TextCNN风格的分类头
        logits = self.classifier(sequence_output, attention_mask)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
        )


# --- 推理类 ---
class BatchCitationClassifier:
    """
    一个封装了模型加载和批量预测逻辑的推理类。
    支持标准BERT分类头、FastText分类头和TextCNN分类头三种架构。
    """

    def __init__(self, model_path: str):
        """
        初始化分类器。

        Args:
            model_path (str): 保存好的模型目录路径。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: '{model_path}'。请确保路径正确。")

        print(f"正在从 '{model_path}' 加载模型和分词器...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 检查模型配置以确定模型类型
        self.model_config = self._load_model_config(model_path)
        
        # 兼容新旧配置格式
        if 'classifier_head_type' in self.model_config:
            # 新格式
            self.classifier_head_type = self.model_config.get('classifier_head_type', 'standard').lower()
        elif 'use_fasttext_head' in self.model_config:
            # 旧格式兼容
            if self.model_config.get('use_fasttext_head', False):
                self.classifier_head_type = 'fasttext'
            else:
                self.classifier_head_type = 'standard'
        else:
            # 默认格式
            self.classifier_head_type = 'standard'
        
        # 根据配置加载相应的模型
        if self.classifier_head_type == 'fasttext':
            print("检测到FastText风格分类头，正在加载自定义模型...")
            self.model = self._load_fasttext_model(model_path)
        elif self.classifier_head_type == 'textcnn':
            print("检测到TextCNN风格分类头，正在加载自定义模型...")
            self.model = self._load_textcnn_model(model_path)
        elif self.classifier_head_type == 'standard':
            print("检测到标准BERT分类头，正在加载标准模型...")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print(f"未知的分类头类型: {self.classifier_head_type}，尝试加载标准模型...")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

        # 从模型的配置中加载标签映射和最大长度
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'id2label'):
                self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            else:
                # 从model_config.json中获取
                self.id2label = self.model_config.get('id2label', {0: 'Primary', 1: 'Secondary', 2: 'Unknown'})
            self.max_length = getattr(self.model.config, 'max_position_embeddings', 512)
        else:
            # 从model_config.json中获取
            self.id2label = self.model_config.get('id2label', {0: 'Primary', 1: 'Secondary', 2: 'Unknown'})
            self.max_length = 512

        # 确保id2label的键为字符串格式（用于查找）
        self.id2label_str = {str(k): v for k, v in self.id2label.items()}

        print(f"模型加载成功，使用设备: {self.device}")
        print(f"模型类型: {self.classifier_head_type.upper()}分类头")
        print(f"标签映射: {self.id2label}")

    def _load_model_config(self, model_path: str) -> Dict[str, Any]:
        """
        加载模型配置文件
        """
        config_path = os.path.join(model_path, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("未找到model_config.json，假设使用标准BERT分类头")
            return {'classifier_head_type': 'standard'}
    
    def _load_fasttext_model(self, model_path: str):
        """
        加载FastText风格的模型
        """
        # 获取模型配置
        num_labels = self.model_config.get('num_labels', 3)
        classifier_dropout = self.model_config.get('classifier_dropout', 0.1)
        
        # 创建模型实例
        # 直接使用模型路径，这样可以从保存的模型中加载BERT权重
        model_name = model_path
        
        # 创建模型
        model = SciBertWithFastTextHead(
            model_name=model_name,
            num_labels=num_labels,
            classifier_dropout=classifier_dropout
        )
        
        # 加载权重
        self._load_model_weights(model, model_path)
        
        return model
    
    def _load_textcnn_model(self, model_path: str):
        """
        加载TextCNN风格的模型
        """
        # 获取模型配置
        num_labels = self.model_config.get('num_labels', 3)
        classifier_dropout = self.model_config.get('classifier_dropout', 0.1)
        filter_sizes = self.model_config.get('textcnn_filter_sizes', [3, 4, 5])
        num_filters = self.model_config.get('textcnn_num_filters', 128)
        
        # 创建模型实例
        # 直接使用模型路径，这样可以从保存的模型中加载BERT权重
        model_name = model_path
        
        # 创建模型
        model = SciBertWithTextCNNHead(
            model_name=model_name,
            num_labels=num_labels,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            classifier_dropout=classifier_dropout
        )
        
        # 加载权重
        self._load_model_weights(model, model_path)
        
        return model
    
    def _load_model_weights(self, model, model_path: str):
        """
        通用的模型权重加载函数
        """
        # 尝试加载pytorch_model.bin或model.safetensors
        weight_files = ['pytorch_model.bin', 'model.safetensors']
        weight_loaded = False
        
        for weight_file in weight_files:
            weight_path = os.path.join(model_path, weight_file)
            if os.path.exists(weight_path):
                print(f"正在加载权重文件: {weight_file}")
                try:
                    if weight_file.endswith('.bin'):
                        state_dict = torch.load(weight_path, map_location='cpu')
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        # safetensors format
                        from safetensors.torch import load_file
                        state_dict = load_file(weight_path)
                        model.load_state_dict(state_dict, strict=False)
                    weight_loaded = True
                    print(f"成功加载权重文件: {weight_file}")
                    break
                except Exception as e:
                    print(f"加载权重文件 {weight_file} 时出错: {e}")
                    continue
        
        if not weight_loaded:
            raise FileNotFoundError(f"在 {model_path} 中未找到可加载的权重文件 (pytorch_model.bin 或 model.safetensors)")

    def _mark_citation(self, chunk: str, dataset_id: str) -> str:
        """
        私有辅助函数，在文本块中标记引用。
        """
        if not dataset_id or dataset_id not in chunk:
            return chunk
            # 情况1：citation_id 以 'https://doi.org/' 开头
        elif dataset_id.startswith('https://doi.org/'):
            doi_without_prefix = dataset_id.replace('https://doi.org/', '', 1)

            # 编译正则表达式（忽略大小写）
            pattern_full = re.compile(re.escape(dataset_id), re.IGNORECASE)
            pattern_doi = re.compile(re.escape(doi_without_prefix), re.IGNORECASE)

            # 优先替换完整的链接（不区分大小写）
            if pattern_full.search(chunk):
                marked_chunk = pattern_full.sub("<cite>DOI</cite>", chunk)
            # 替换不带前缀的DOI（不区分大小写）
            elif pattern_doi.search(chunk):
                marked_chunk = pattern_doi.sub("<cite>DOI</cite>", chunk)
            else:
                marked_chunk = chunk

        # 情况2：其他类型的ID（如 Accession ID）
        else:
            pattern_id = re.compile(re.escape(dataset_id), re.IGNORECASE)
            if pattern_id.search(chunk):
                marked_chunk = pattern_id.sub("<cite>Accession IDs</cite>", chunk)
            else:
                marked_chunk = chunk
        url_pattern = re.compile(r'https?://[^\s<>"]+')

        # 使用 re.sub 将所有匹配到的链接替换为空字符串 ""
        final_chunk = re.sub(url_pattern, '', marked_chunk)

        return final_chunk

    def predict_batch(self, chunks_data: List[Tuple[Any, str, str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        对一批数据进行分批预测，以防止显存溢出。

        Args:
            chunks_data (List[Tuple]): 格式为 (article_id, chunk, result_value, pattern_name) 的元组列表。
            batch_size (int): 每个小批次的大小。如果仍然OOM，请减小此值。

        Returns:
            List[Dict]: 包含原始数据和预测结果的字典列表。
        """
        if not chunks_data:
            print("[Warning] 输入数据为空，返回空列表。")
            return []

        print(f"开始对 {len(chunks_data)} 条数据进行批量预测，内部批次大小为 {batch_size}...")
        print(f"使用模型类型: {self.classifier_head_type.upper()}分类头")
        
        all_results = []
        # 使用 tqdm 创建一个带进度条的循环
        for i in tqdm(range(0, len(chunks_data), batch_size), desc="推理进度"):
            # 1. 获取一个小批次的数据
            batch_chunk_data = chunks_data[i : i + batch_size]
            
            # 2. 预处理当前批次的文本
            texts_to_process = [
                self._mark_citation(chunk=item[1], dataset_id=item[2])
                for item in batch_chunk_data
            ]
            
            # 3. 批量分词
            inputs = self.tokenizer(
                texts_to_process,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            # 4. 模型预测
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 5. 后处理结果
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_ids = torch.argmax(probabilities, dim=-1)

            # 6. 整理当前批次的结果并添加到总结果列表中
            for j, original_item in enumerate(batch_chunk_data):
                predicted_id = predicted_class_ids[j].item()
                result_dict = {
                    'article_id': original_item[0],
                    'chunk': original_item[1],
                    'citation_id': original_item[2],
                    'pattern_name': original_item[3],
                    'processed_text': texts_to_process[j],
                    'predicted_label': self.id2label_str.get(str(predicted_id), self.id2label.get(predicted_id, 'Unknown')),
                    'confidence': probabilities[j][predicted_id].item(),
                    'all_probabilities': {
                        self.id2label_str.get(str(label_id), self.id2label.get(label_id, f'Label_{label_id}')): prob.item()
                        for label_id, prob in enumerate(probabilities[j])
                    }
                }
                all_results.append(result_dict)
                
            # (可选，但推荐) 清理CUDA缓存，释放未使用的显存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        print("批量预测完成。")
        return all_results

    def predict_single(self, chunk: str, citation_id: str) -> Dict[str, Any]:
        """
        对单个样本进行预测
        
        Args:
            chunk (str): 文本片段
            citation_id (str): 引用ID
            
        Returns:
            Dict: 预测结果
        """
        # 预处理文本
        processed_text = self._mark_citation(chunk, citation_id)
        
        # 分词
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 后处理
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        
        result = {
            'chunk': chunk,
            'citation_id': citation_id,
            'processed_text': processed_text,
            'predicted_label': self.id2label_str.get(str(predicted_class_id), self.id2label.get(predicted_class_id, 'Unknown')),
            'confidence': probabilities[0][predicted_class_id].item(),
            'all_probabilities': {
                self.id2label_str.get(str(label_id), self.id2label.get(label_id, f'Label_{label_id}')): prob.item()
                for label_id, prob in enumerate(probabilities[0])
            }
        }
        
        return result


def load_model_for_inference(model_path: str):
    """
    便捷函数：加载模型用于推理
    
    Args:
        model_path (str): 模型路径
        
    Returns:
        BatchCitationClassifier: 推理分类器实例
    """
    return BatchCitationClassifier(model_path)