#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集查找器 - 通过DOI查找相关数据集
基于Crossref和DataCite API
支持多进程并发处理
"""

import requests
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote
import multiprocessing as mp
from multiprocessing import Pool, Manager
import os
from functools import partial

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_finder.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _process_single_article_worker(article: Dict, delay: float = 1.0, max_concurrent_requests: int = 5) -> Tuple[str, List[str]]:
    """
    处理单篇文章的工作函数（用于多进程）
    
    Args:
        article: 文章信息
        delay: 请求间隔时间
        max_concurrent_requests: 最大并发请求数
        
    Returns:
        (article_doi_key, datasets_list) 元组
    """
    try:
        # 获取文章DOI
        doi = article.get('doi', '')
        if not doi:
            logger.warning(f"文章没有DOI，跳过: {article.get('title', 'Unknown')}")
            return "", []
        
        # 创建临时的查找器实例
        temp_finder = DatasetFinder(delay=delay, max_concurrent_requests=max_concurrent_requests)
        
        # 查找数据集
        datasets = temp_finder.find_datasets_for_article(doi, max_concurrent_requests)
        
        if datasets:
            # 标准化DOI作为key
            key = temp_finder.normalize_doi_key(doi)
            logger.info(f"文章 {doi} 找到 {len(datasets)} 个数据集")
            return key, datasets
        else:
            logger.info(f"文章 {doi} 没有找到相关数据集")
            return "", []
            
    except Exception as e:
        logger.error(f"处理文章时出错: {e}")
        return "", []

def _process_single_article_worker_wrapper(args):
    """包装函数，用于多进程调用"""
    if len(args) == 3:
        article, delay, max_concurrent_requests = args
        return _process_single_article_worker(article, delay, max_concurrent_requests)
    else:
        article, delay = args
        return _process_single_article_worker(article, delay, 5)  # 默认值

class DatasetFinder:
    """数据集查找器 - 通过DOI查找相关数据集"""
    
    def __init__(self, delay: float = 1.0, max_retries: int = 5, max_workers: int = None, max_concurrent_requests: int = 5):
        """
        初始化数据集查找器
        
        Args:
            delay: 请求间隔时间（秒）
            max_retries: 最大重试次数（默认5次，403/429错误会使用更多重试）
            max_workers: 最大工作进程数（None表示使用CPU核心数）
            max_concurrent_requests: 最大并发请求数（用于检查数据集）
        """
        self.delay = delay
        self.max_retries = max_retries
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # 限制最大进程数
        self.max_concurrent_requests = max_concurrent_requests
        
        # 针对不同错误类型的重试策略
        self.retry_strategies = {
            '403': {
                'max_retries': max_retries + 2,  # 403错误增加2次重试
                'backoff_factor': 3,  # 3的幂次方
                'description': '请求频率过高'
            },
            '429': {
                'max_retries': max_retries + 3,  # 429错误增加3次重试
                'backoff_factor': 4,  # 4的幂次方
                'description': '请求过多'
            },
            'default': {
                'max_retries': max_retries,
                'backoff_factor': 1.5,  # 1.5的幂次方
                'description': '一般错误'
            }
        }
        
        # 设置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # 预过滤规则：明显不是数据集的DOI模式
        self.non_dataset_patterns = [
            # 期刊DOI模式（应该被过滤）
            "10.1111/j.",  # Wiley期刊
            "10.1016/j.",  # Elsevier期刊
            "10.1002/",    # Wiley期刊
            "10.1038/",    # Nature期刊
            "10.1126/",    # Science期刊
            "10.1073/",    # PNAS期刊
            "10.1093/",    # Oxford期刊
            "10.1103/",    # APS期刊
            "10.1021/",    # ACS期刊
            "10.1007/",    # Springer期刊
            "10.1186/",    # BioMed Central期刊
            "10.1371/",    # PLOS期刊
            "10.3390/",    # MDPI期刊
            "10.1155/",    # Hindawi期刊
            "10.1159/",    # Karger期刊
            "10.1161/",    # AHA期刊
            "10.1210/",    # Endocrine Society期刊
            "10.1212/",    # AAN期刊
            "10.1214/",    # IMS期刊
            "10.1215/",    # Duke University Press期刊
            "10.1242/",    # Company of Biologists期刊
            "10.1256/",    # Royal Meteorological Society期刊
            "10.1261/",    # Cold Spring Harbor Laboratory期刊
            "10.1289/",    # Environmental Health Perspectives期刊
            "10.1299/",    # JSME期刊
            "10.1300/",    # Haworth Press期刊
            "10.1310/",    # SAGE期刊
            "10.1320/",    # 其他期刊
            "10.1330/",    # 其他期刊
            "10.1340/",    # 其他期刊
            "10.1350/",    # 其他期刊
            "10.1360/",    # 其他期刊
            "10.1370/",    # 其他期刊
            "10.1380/",    # 其他期刊
            "10.1390/",    # 其他期刊
            "10.1400/",    # 其他期刊
            "10.1410/",    # 其他期刊
            "10.1420/",    # 其他期刊
            "10.1430/",    # 其他期刊
            "10.1440/",    # 其他期刊
            "10.1450/",    # 其他期刊
            "10.1460/",    # 其他期刊
            "10.1470/",    # 其他期刊
            "10.1480/",    # 其他期刊
            "10.1490/",    # 其他期刊
            "10.1500/",    # 其他期刊
            "10.1510/",    # 其他期刊
            "10.1520/",    # 其他期刊
            "10.1530/",    # 其他期刊
            "10.1540/",    # 其他期刊
            "10.1550/",    # 其他期刊
            "10.1560/",    # 其他期刊
            "10.1570/",    # 其他期刊
            "10.1580/",    # 其他期刊
            "10.1590/",    # 其他期刊
            "10.1600/",    # 其他期刊
            "10.1610/",    # 其他期刊
            "10.1620/",    # 其他期刊
            "10.1630/",    # 其他期刊
            "10.1640/",    # 其他期刊
            "10.1650/",    # 其他期刊
            "10.1660/",    # 其他期刊
            "10.1670/",    # 其他期刊
            "10.1680/",    # 其他期刊
            "10.1690/",    # 其他期刊
            "10.1700/",    # 其他期刊
            "10.1710/",    # 其他期刊
            "10.1720/",    # 其他期刊
            "10.1730/",    # 其他期刊
            "10.1740/",    # 其他期刊
            "10.1750/",    # 其他期刊
            "10.1760/",    # 其他期刊
            "10.1770/",    # 其他期刊
            "10.1780/",    # 其他期刊
            "10.1790/",    # 其他期刊
            "10.1800/",    # 其他期刊
            "10.1810/",    # 其他期刊
            "10.1820/",    # 其他期刊
            "10.1830/",    # 其他期刊
            "10.1840/",    # 其他期刊
            "10.1850/",    # 其他期刊
            "10.1860/",    # 其他期刊
            "10.1870/",    # 其他期刊
            "10.1880/",    # 其他期刊
            "10.1890/",    # 其他期刊
            "10.1900/",    # 其他期刊
            "10.1910/",    # 其他期刊
            "10.1920/",    # 其他期刊
            "10.1930/",    # 其他期刊
            "10.1940/",    # 其他期刊
            "10.1950/",    # 其他期刊
            "10.1960/",    # 其他期刊
            "10.1970/",    # 其他期刊
            "10.1980/",    # 其他期刊
            "10.1990/",    # 其他期刊
        ]
        
        # 数据集白名单：明显是数据集的DOI模式（直接保存，不请求API）
        self.dataset_whitelist_patterns = [
            "10.5281/",    # Zenodo数据集
            "10.5061/",    # Dryad数据集
            "10.7910/",    # Harvard Dataverse数据集
            "10.6073/",    # ICPSR数据集
            "10.6084/",    # Figshare数据集
            "10.4121/",    # 4TU.ResearchData数据集
            "10.17026/",   # DANS数据集
            "10.24432/",   # 其他数据集
            "10.24433/",   # 其他数据集
            "10.24434/",   # 其他数据集
            "10.24435/",   # 其他数据集
            "10.24436/",   # 其他数据集
            "10.24437/",   # 其他数据集
            "10.24438/",   # 其他数据集
            "10.24439/",   # 其他数据集
            "10.24440/",   # 其他数据集
            "10.24441/",   # 其他数据集
            "10.24442/",   # 其他数据集
            "10.24443/",   # 其他数据集
            "10.24444/",   # 其他数据集
            "10.24445/",   # 其他数据集
            "10.24446/",   # 其他数据集
            "10.24447/",   # 其他数据集
            "10.24448/",   # 其他数据集
            "10.24449/",   # 其他数据集
            "10.24450/",   # 其他数据集
            "10.24451/",   # 其他数据集
            "10.24452/",   # 其他数据集
            "10.24453/",   # 其他数据集
            "10.24454/",   # 其他数据集
            "10.24455/",   # 其他数据集
            "10.24456/",   # 其他数据集
            "10.24457/",   # 其他数据集
            "10.24458/",   # 其他数据集
            "10.24459/",   # 其他数据集
            "10.24460/",   # 其他数据集
            "10.24461/",   # 其他数据集
            "10.24462/",   # 其他数据集
            "10.24463/",   # 其他数据集
            "10.24464/",   # 其他数据集
            "10.24465/",   # 其他数据集
            "10.24466/",   # 其他数据集
            "10.24467/",   # 其他数据集
            "10.24468/",   # 其他数据集
            "10.24469/",   # 其他数据集
            "10.24470/",   # 其他数据集
            "10.24471/",   # 其他数据集
            "10.24472/",   # 其他数据集
            "10.24473/",   # 其他数据集
            "10.24474/",   # 其他数据集
            "10.24475/",   # 其他数据集
            "10.24476/",   # 其他数据集
            "10.24477/",   # 其他数据集
            "10.24478/",   # 其他数据集
            "10.24479/",   # 其他数据集
            "10.24480/",   # 其他数据集
            "10.24481/",   # 其他数据集
            "10.24482/",   # 其他数据集
            "10.24483/",   # 其他数据集
            "10.24484/",   # 其他数据集
            "10.24485/",   # 其他数据集
            "10.24486/",   # 其他数据集
            "10.24487/",   # 其他数据集
            "10.24488/",   # 其他数据集
            "10.24489/",   # 其他数据集
            "10.24490/",   # 其他数据集
            "10.24491/",   # 其他数据集
            "10.24492/",   # 其他数据集
            "10.24493/",   # 其他数据集
            "10.24494/",   # 其他数据集
            "10.24495/",   # 其他数据集
            "10.24496/",   # 其他数据集
            "10.24497/",   # 其他数据集
            "10.24498/",   # 其他数据集
            "10.24499/",   # 其他数据集
        ]
        
        # 统计信息
        self.stats = {
            'total_articles': 0,
            'articles_with_datasets': 0,
            'total_datasets_found': 0,
            'crossref_requests': 0,
            'datacite_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,  # 新增：记录API限制次数
            'retry_attempts': 0,   # 新增：记录重试次数
            'total_wait_time': 0,  # 新增：记录总等待时间
            'filtered_dois': 0,    # 新增：记录被过滤的DOI数量
            'pre_filtered_dois': 0, # 新增：记录预过滤的DOI数量
            'whitelist_datasets': 0, # 新增：记录白名单直接保存的数据集数量
            'api_checked_datasets': 0 # 新增：记录通过API检查的数据集数量
        }
    
    def normalize_doi_key(self, doi: str) -> str:
        """
        将DOI标准化为JSON的key（将斜杠替换为下划线）
        
        Args:
            doi: DOI字符串
            
        Returns:
            标准化的key
        """
        if not doi:
            return ""
        
        # 移除DOI前缀
        if doi.startswith('https://doi.org/'):
            doi = doi[18:]
        elif doi.startswith('http://doi.org/'):
            doi = doi[17:]
        
        # 将斜杠替换为下划线
        return doi.replace('/', '_')
    
    def query_crossref_references(self, doi: str) -> List[str]:
        """
        查询Crossref获取参考文献DOI列表
        
        Args:
            doi: 文章DOI
            
        Returns:
            参考文献DOI列表
        """
        try:
            logger.info(f"查询Crossref参考文献: {doi}")
            
            # 创建新的会话（每个进程独立）
            session = requests.Session()
            session.headers.update(self.headers)
            
            # 构建Crossref API URL
            encoded_doi = quote(doi, safe='')
            url = f"https://api.crossref.org/works/{encoded_doi}"
            
            # 发送请求
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            # 解析JSON响应
            data = response.json()
            
            # 提取参考文献DOI
            references = []
            if 'message' in data and 'reference' in data['message']:
                reference_list = data['message']['reference']
                for ref in reference_list:
                    if 'DOI' in ref and ref['DOI']:
                        references.append(ref['DOI'])
            
            logger.info(f"从Crossref获取到 {len(references)} 个参考文献DOI")
            return references
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Crossref API请求失败: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Crossref API响应解析失败: {e}")
            return []
        except Exception as e:
            logger.error(f"查询Crossref参考文献时出错: {e}")
            return []
    
    def check_datacite_dataset(self, doi: str) -> bool:
        """
        检查DataCite中是否存在该DOI（数据集）
        
        Args:
            doi: DOI字符串
            
        Returns:
            是否为数据集
        """
        base_delay = self.delay
        
        for attempt in range(self.max_retries + 3 + 1):  # 最大重试次数（包括429的额外重试）
            try:
                logger.debug(f"检查DataCite数据集: {doi} (尝试 {attempt + 1}/{self.max_retries + 3 + 1})")
                
                # 创建新的会话（每个进程独立）
                session = requests.Session()
                session.headers.update(self.headers)
                
                # 构建DataCite API URL
                encoded_doi = quote(doi, safe='')
                url = f"https://api.datacite.org/dois/{encoded_doi}"
                
                # 发送请求
                response = session.get(url, timeout=30)
                
                # 检查响应状态
                if response.status_code == 200:
                    # 成功返回，说明是数据集
                    data = response.json()
                    # 可以进一步检查数据类型
                    if 'data' in data and 'attributes' in data['data']:
                        resource_type = data['data']['attributes'].get('resourceType', '')
                        if 'dataset' in resource_type.lower():
                            logger.debug(f"确认是数据集: {doi}")
                            return True
                    return True  # 如果DataCite返回成功，通常就是数据集
                elif response.status_code == 404:
                    # 不存在，不是数据集
                    return False
                elif response.status_code == 403:
                    # 请求频率过高，使用专门的403重试策略
                    strategy = self.retry_strategies['403']
                    self.stats['rate_limit_hits'] += 1
                    if attempt < strategy['max_retries']:
                        self.stats['retry_attempts'] += 1
                        wait_time = base_delay * (strategy['backoff_factor'] ** attempt)
                        self.stats['total_wait_time'] += wait_time
                        logger.warning(f"DataCite API返回403，{strategy['description']}，等待 {wait_time} 秒后重试: {doi}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"DataCite API持续返回403，已达到最大重试次数({strategy['max_retries']}): {doi}")
                        return False
                elif response.status_code == 429:
                    # 请求过多，使用专门的429重试策略
                    strategy = self.retry_strategies['429']
                    self.stats['rate_limit_hits'] += 1
                    if attempt < strategy['max_retries']:
                        self.stats['retry_attempts'] += 1
                        wait_time = base_delay * (strategy['backoff_factor'] ** attempt)
                        self.stats['total_wait_time'] += wait_time
                        logger.warning(f"DataCite API返回429，{strategy['description']}，等待 {wait_time} 秒后重试: {doi}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"DataCite API持续返回429，已达到最大重试次数({strategy['max_retries']}): {doi}")
                        return False
                else:
                    # 其他错误状态，使用默认重试策略
                    strategy = self.retry_strategies['default']
                    if attempt < strategy['max_retries']:
                        wait_time = base_delay * (strategy['backoff_factor'] ** attempt)
                        logger.warning(f"DataCite API返回状态码 {response.status_code}，等待 {wait_time} 秒后重试: {doi}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"DataCite API返回状态码 {response.status_code}，已达到最大重试次数({strategy['max_retries']}): {doi}")
                        return False
                
            except requests.exceptions.RequestException as e:
                strategy = self.retry_strategies['default']
                if attempt < strategy['max_retries']:
                    wait_time = base_delay * (strategy['backoff_factor'] ** attempt)
                    logger.warning(f"DataCite API请求失败，等待 {wait_time} 秒后重试: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"DataCite API请求失败，已达到最大重试次数({strategy['max_retries']}): {e}")
                    return False
            except json.JSONDecodeError as e:
                strategy = self.retry_strategies['default']
                if attempt < strategy['max_retries']:
                    wait_time = base_delay * (strategy['backoff_factor'] ** attempt)
                    logger.warning(f"DataCite API响应解析失败，等待 {wait_time} 秒后重试: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"DataCite API响应解析失败，已达到最大重试次数({strategy['max_retries']}): {e}")
                    return False
            except Exception as e:
                strategy = self.retry_strategies['default']
                if attempt < strategy['max_retries']:
                    wait_time = base_delay * (strategy['backoff_factor'] ** attempt)
                    logger.warning(f"检查DataCite数据集时出错，等待 {wait_time} 秒后重试: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"检查DataCite数据集时出错，已达到最大重试次数({strategy['max_retries']}): {e}")
                    return False
        
        return False
    
    def find_datasets_for_article(self, article_doi: str, max_concurrent_requests: int = 5) -> List[str]:
        """
        为单篇文章查找相关数据集
        
        Args:
            article_doi: 文章DOI
            max_concurrent_requests: 最大并发请求数
            
        Returns:
            相关数据集DOI列表
        """
        if not article_doi:
            return []
        
        logger.info(f"开始为文章查找数据集: {article_doi}")
        
        # 查询Crossref获取参考文献
        reference_dois = self.query_crossref_references(article_doi)
        
        if not reference_dois:
            logger.info(f"文章 {article_doi} 没有找到参考文献")
            return []
        
        # 预过滤DOI
        api_check_dois, whitelist_datasets = self.pre_filter_dois(reference_dois)
        
        # 使用并发请求检查参考文献是否为数据集
        api_datasets = self._check_datasets_concurrent(api_check_dois, max_concurrent_requests)
        
        # 合并白名单数据集和API检查结果
        all_datasets = whitelist_datasets + api_datasets
        
        logger.info(f"文章 {article_doi} 找到 {len(all_datasets)} 个数据集")
        logger.info(f"  - 白名单数据集: {len(whitelist_datasets)} 个")
        logger.info(f"  - API检查数据集: {len(api_datasets)} 个")
        
        return all_datasets
    
    def pre_filter_dois(self, reference_dois: List[str]) -> Tuple[List[str], List[str]]:
        """
        预过滤DOI列表，返回需要API检查的DOI和直接保存的数据集DOI
        
        Args:
            reference_dois: 参考文献DOI列表
            
        Returns:
            (需要API检查的DOI列表, 直接保存的数据集DOI列表)
        """
        if not reference_dois:
            return [], []
        
        original_count = len(reference_dois)
        api_check_dois = []
        whitelist_datasets = []
        
        for doi in reference_dois:
            if self._is_in_whitelist(doi):
                # 白名单数据集，直接保存
                whitelist_datasets.append(doi)
                self.stats['whitelist_datasets'] += 1
                logger.debug(f"白名单数据集（直接保存）: {doi}")
            elif self._is_likely_dataset(doi):
                # 可能是数据集，需要API检查
                api_check_dois.append(doi)
                logger.debug(f"需要API检查的DOI: {doi}")
            else:
                # 明显不是数据集，过滤掉
                self.stats['pre_filtered_dois'] += 1
                logger.debug(f"预过滤DOI（明显不是数据集）: {doi}")
        
        filtered_count = len(api_check_dois)
        whitelist_count = len(whitelist_datasets)
        removed_count = original_count - filtered_count - whitelist_count
        
        logger.info(f"预过滤完成: {original_count} -> API检查: {filtered_count}, 白名单: {whitelist_count}, 过滤: {removed_count}")
        
        return api_check_dois, whitelist_datasets
    
    def _is_in_whitelist(self, doi: str) -> bool:
        """
        检查DOI是否在白名单中（明显是数据集）
        
        Args:
            doi: DOI字符串
            
        Returns:
            是否在白名单中
        """
        if not doi:
            return False
        
        # 移除DOI前缀
        if doi.startswith('https://doi.org/'):
            doi = doi[18:]
        elif doi.startswith('http://doi.org/'):
            doi = doi[17:]
        
        # 检查是否匹配白名单模式
        for pattern in self.dataset_whitelist_patterns:
            if doi.startswith(pattern):
                return True
        
        return False
    
    def _is_likely_dataset(self, doi: str) -> bool:
        """
        判断DOI是否可能是数据集
        
        Args:
            doi: DOI字符串
            
        Returns:
            是否可能是数据集
        """
        if not doi:
            return False
        
        # 移除DOI前缀
        if doi.startswith('https://doi.org/'):
            doi = doi[18:]
        elif doi.startswith('http://doi.org/'):
            doi = doi[17:]
        
        # 检查是否匹配非数据集模式
        for pattern in self.non_dataset_patterns:
            if doi.startswith(pattern):
                return False
        
        # 检查是否包含明显的期刊关键词
        journal_keywords = [
            'journal', 'article', 'paper', 'research', 'study', 'review',
            'letter', 'note', 'comment', 'correspondence', 'editorial',
            'abstract', 'proceedings', 'conference', 'symposium', 'workshop'
        ]
        
        doi_lower = doi.lower()
        for keyword in journal_keywords:
            if keyword in doi_lower:
                return False
        
        # 检查DOI结构（数据集通常有特定的结构）
        # 数据集DOI通常包含数字和字母的组合，长度适中
        if len(doi) < 10 or len(doi) > 100:
            return False
        
        # 数据集DOI通常包含斜杠分隔的路径结构
        if '/' not in doi:
            return False
        
        # 数据集DOI通常不以常见的期刊标识符结尾
        journal_suffixes = [
            '.pdf', '.html', '.xml', '.txt', '.doc', '.docx',
            'abstract', 'full', 'supplement', 'appendix'
        ]
        
        for suffix in journal_suffixes:
            if doi.lower().endswith(suffix):
                return False
        
        return True
    
    def _check_datasets_concurrent(self, reference_dois: List[str], max_concurrent: int = 5) -> List[str]:
        """
        并发检查参考文献是否为数据集
        
        Args:
            reference_dois: 参考文献DOI列表
            max_concurrent: 最大并发请求数
            
        Returns:
            数据集DOI列表
        """
        if not reference_dois:
            return []
        
        # 更新API检查统计
        self.stats['api_checked_datasets'] += len(reference_dois)
        
        # 如果参考文献数量较少，使用顺序处理
        if len(reference_dois) <= 3:
            return self._check_datasets_sequential(reference_dois)
        
        logger.info(f"使用并发请求检查 {len(reference_dois)} 个参考文献（最大并发数: {max_concurrent}）")
        
        # 使用线程池进行并发请求
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        datasets = []
        lock = threading.Lock()
        error_count = 0
        current_concurrent = max_concurrent
        
        def check_single_dataset(ref_doi):
            """检查单个DOI是否为数据集"""
            nonlocal error_count, current_concurrent
            try:
                if self.check_datacite_dataset(ref_doi):
                    with lock:
                        datasets.append(ref_doi)
                        logger.info(f"找到数据集: {ref_doi}")
                    return ref_doi
                return None
            except Exception as e:
                with lock:
                    error_count += 1
                    logger.error(f"检查数据集时出错 {ref_doi}: {e}")
                return None
        
        # 分批处理，动态调整并发数
        batch_size = min(10, len(reference_dois))  # 每批最多10个
        remaining_dois = reference_dois.copy()
        
        while remaining_dois:
            # 动态调整并发数
            if error_count > 5:
                current_concurrent = max(1, current_concurrent // 2)
                logger.warning(f"错误较多，减少并发数到: {current_concurrent}")
                error_count = 0  # 重置错误计数
            
            # 取出一批DOI
            batch_dois = remaining_dois[:batch_size]
            remaining_dois = remaining_dois[batch_size:]
            
            logger.info(f"处理批次: {len(batch_dois)} 个DOI，当前并发数: {current_concurrent}")
            
            # 使用线程池执行并发请求
            with ThreadPoolExecutor(max_workers=current_concurrent) as executor:
                # 提交批次任务
                future_to_doi = {executor.submit(check_single_dataset, ref_doi): ref_doi 
                               for ref_doi in batch_dois}
                
                # 收集结果
                for future in as_completed(future_to_doi):
                    ref_doi = future_to_doi[future]
                    try:
                        result = future.result()
                        # 结果已经在check_single_dataset中处理了
                    except Exception as e:
                        with lock:
                            error_count += 1
                        logger.error(f"处理 {ref_doi} 时出错: {e}")
            
            # 批次间延迟，避免请求过于密集
            if remaining_dois:
                batch_delay = self.delay * 2  # 批次间延迟加倍
                logger.info(f"批次完成，等待 {batch_delay} 秒后处理下一批...")
                time.sleep(batch_delay)
        
        logger.info(f"并发检查完成，找到 {len(datasets)} 个数据集")
        return datasets
    
    def _check_datasets_sequential(self, reference_dois: List[str]) -> List[str]:
        """
        顺序检查参考文献是否为数据集（原始方法）
        
        Args:
            reference_dois: 参考文献DOI列表
            
        Returns:
            数据集DOI列表
        """
        datasets = []
        for ref_doi in reference_dois:
            if self.check_datacite_dataset(ref_doi):
                datasets.append(ref_doi)
                logger.info(f"找到数据集: {ref_doi}")
            
            # 请求间隔
            time.sleep(self.delay)
        
        return datasets
    
    def process_articles_parallel(self, articles: List[Dict]) -> Dict[str, List[str]]:
        """
        使用多进程并发处理文章列表，查找相关数据集
        
        Args:
            articles: 文章信息列表
            
        Returns:
            数据集映射字典 {article_doi_key: [dataset_dois]}
        """
        self.stats['total_articles'] = len(articles)
        datasets_by_article = {}
        
        logger.info(f"开始使用 {self.max_workers} 个进程并发处理 {len(articles)} 篇文章")
        
        # 使用进程池
        with Pool(processes=self.max_workers) as pool:
            # 准备参数，包含max_concurrent_requests
            args_list = [(article, self.delay, self.max_concurrent_requests) for article in articles]
            
            # 使用imap处理文章，保持顺序
            results = pool.imap(_process_single_article_worker_wrapper, args_list)
            
            # 收集结果
            for i, (key, datasets) in enumerate(results, 1):
                logger.info(f"完成第 {i}/{len(articles)} 篇文章的处理")
                
                if key and datasets:
                    datasets_by_article[key] = datasets
                    self.stats['articles_with_datasets'] += 1
                    self.stats['total_datasets_found'] += len(datasets)
        
        # 打印统计信息
        self.print_stats()
        
        return datasets_by_article
    
    def process_articles(self, articles: List[Dict]) -> Dict[str, List[str]]:
        """
        处理文章列表，查找相关数据集（兼容性方法）
        
        Args:
            articles: 文章信息列表
            
        Returns:
            数据集映射字典 {article_doi_key: [dataset_dois]}
        """
        # 如果文章数量较少，使用单进程处理
        if len(articles) <= 5:
            logger.info("文章数量较少，使用单进程处理")
            return self._process_articles_sequential(articles)
        else:
            logger.info("文章数量较多，使用多进程并发处理")
            return self.process_articles_parallel(articles)
    
    def _process_articles_sequential(self, articles: List[Dict]) -> Dict[str, List[str]]:
        """
        单进程顺序处理文章列表（原始方法）
        
        Args:
            articles: 文章信息列表
            
        Returns:
            数据集映射字典 {article_doi_key: [dataset_dois]}
        """
        self.stats['total_articles'] = len(articles)
        datasets_by_article = {}
        
        logger.info(f"开始顺序处理 {len(articles)} 篇文章")
        
        for i, article in enumerate(articles, 1):
            try:
                logger.info(f"处理第 {i}/{len(articles)} 篇文章")
                
                # 获取文章DOI
                doi = article.get('doi', '')
                if not doi:
                    logger.warning(f"文章没有DOI，跳过: {article.get('title', 'Unknown')}")
                    continue
                
                # 查找数据集，使用max_concurrent_requests参数
                datasets = self.find_datasets_for_article(doi, self.max_concurrent_requests)
                
                if datasets:
                    # 标准化DOI作为key
                    key = self.normalize_doi_key(doi)
                    datasets_by_article[key] = datasets
                    
                    self.stats['articles_with_datasets'] += 1
                    self.stats['total_datasets_found'] += len(datasets)
                    
                    logger.info(f"文章 {doi} 找到 {len(datasets)} 个数据集")
                else:
                    logger.info(f"文章 {doi} 没有找到相关数据集")
                
                # 请求间隔
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"处理文章时出错: {e}")
                continue
        
        # 打印统计信息
        self.print_stats()
        
        return datasets_by_article
    
    def save_datasets_json(self, datasets_by_article: Dict[str, List[str]], filename: str = "datasets_by_article.json") -> None:
        """
        保存数据集映射到JSON文件
        
        Args:
            datasets_by_article: 数据集映射字典
            filename: 输出文件名
        """
        try:
            # 收集所有唯一的数据集DOI
            all_datasets = set()
            for datasets in datasets_by_article.values():
                all_datasets.update(datasets)
            
            # 创建完整的输出数据结构
            output_data = {
                "datasets_by_article": datasets_by_article,
                "all_datasets": sorted(list(all_datasets)),  # 排序并转换为列表
                "summary": {
                    "total_articles": len(datasets_by_article),
                    "total_unique_datasets": len(all_datasets),
                    "articles_with_datasets": len(datasets_by_article)
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据集映射已保存到: {filename}")
            logger.info(f"共保存 {len(datasets_by_article)} 篇文章的数据集信息")
            logger.info(f"总共找到 {len(all_datasets)} 个唯一的数据集DOI")
            
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
    
    def print_stats(self) -> None:
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("数据集查找统计信息:")
        logger.info(f"总文章数: {self.stats['total_articles']}")
        logger.info(f"找到数据集的文章数: {self.stats['articles_with_datasets']}")
        logger.info(f"总数据集数量: {self.stats['total_datasets_found']}")
        logger.info(f"使用进程数: {self.max_workers}")
        logger.info(f"API限制次数: {self.stats['rate_limit_hits']}")
        logger.info(f"重试次数: {self.stats['retry_attempts']}")
        logger.info(f"总等待时间: {self.stats['total_wait_time']:.1f} 秒")
        logger.info(f"预过滤DOI数量: {self.stats['pre_filtered_dois']}")
        logger.info(f"白名单数据集数量: {self.stats['whitelist_datasets']}")
        logger.info(f"API检查数据集数量: {self.stats['api_checked_datasets']}")
        logger.info(f"过滤DOI数量: {self.stats['filtered_dois']}")
        
        # 显示重试策略信息
        logger.info("-" * 30)
        logger.info("重试策略配置:")
        logger.info(f"默认重试次数: {self.retry_strategies['default']['max_retries']}")
        logger.info(f"403错误重试次数: {self.retry_strategies['403']['max_retries']}")
        logger.info(f"429错误重试次数: {self.retry_strategies['429']['max_retries']}")
        logger.info(f"403错误退避因子: {self.retry_strategies['403']['backoff_factor']}")
        logger.info(f"429错误退避因子: {self.retry_strategies['429']['backoff_factor']}")
        logger.info(f"默认退避因子: {self.retry_strategies['default']['backoff_factor']}")
        
        if self.stats['total_articles'] > 0:
            success_rate = (self.stats['articles_with_datasets'] / self.stats['total_articles']) * 100
            logger.info(f"找到数据集的文章比例: {success_rate:.1f}%")
        
        if self.stats['rate_limit_hits'] > 0:
            logger.info(f"API限制频率: {self.stats['rate_limit_hits'] / self.stats['total_articles']:.2f} 次/文章")
        
        if self.stats['pre_filtered_dois'] > 0:
            logger.info(f"预过滤效率: 过滤掉 {self.stats['pre_filtered_dois']} 个明显不是数据集的DOI")
        
        if self.stats['whitelist_datasets'] > 0:
            logger.info(f"白名单效率: 直接保存 {self.stats['whitelist_datasets']} 个数据集（无需API请求）")
        
        if self.stats['api_checked_datasets'] > 0:
            api_success_rate = (self.stats['total_datasets_found'] - self.stats['whitelist_datasets']) / self.stats['api_checked_datasets'] * 100
            logger.info(f"API检查成功率: {api_success_rate:.1f}%")
        
        logger.info("=" * 50)
    
    def run(self, articles: List[Dict], output_file: str = "datasets_by_article.json") -> None:
        """
        运行数据集查找器
        
        Args:
            articles: 文章信息列表
            output_file: 输出文件名
        """
        logger.info("开始运行数据集查找器")
        logger.info(f"处理文章数量: {len(articles)}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"最大工作进程数: {self.max_workers}")
        
        # 处理文章
        datasets_by_article = self.process_articles(articles)
        
        # 保存结果
        if datasets_by_article:
            self.save_datasets_json(datasets_by_article, output_file)
        else:
            logger.warning("没有找到任何数据集")
        
        logger.info("数据集查找器运行完成")

def main():
    """主函数 - 用于独立测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集查找器 - 通过DOI查找相关数据集')
    parser.add_argument('--input-file', required=True, help='包含文章信息的JSON文件')
    parser.add_argument('--output-file', default='datasets_by_article.json', help='输出文件名')
    parser.add_argument('--delay', type=float, default=1.0, help='请求间隔时间（秒）')
    parser.add_argument('--max-workers', type=int, default=None, help='最大工作进程数（默认使用CPU核心数）')
    parser.add_argument('--max-concurrent-requests', type=int, default=5, help='最大并发请求数（用于检查数据集）')
    
    args = parser.parse_args()
    
    # 读取文章信息
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except Exception as e:
        logger.error(f"读取输入文件失败: {e}")
        return
    
    # 创建数据集查找器
    finder = DatasetFinder(
        delay=args.delay, 
        max_workers=args.max_workers,
        max_concurrent_requests=args.max_concurrent_requests
    )
    
    # 运行查找器
    finder.run(articles, args.output_file)

if __name__ == "__main__":
    main() 