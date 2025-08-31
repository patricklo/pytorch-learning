import os
import pandas as pd
LOCAL = sum(['KAGGLE' in k for k in os.environ]) == 0
if not LOCAL:
    # !mkdir -p /root/.cache/datalab/models/
    # !mkdir -p /usr/local/lib/python3.11/dist-packages/static/fonts/
    # !cp -r /kaggle/input/marker-models/marker_models/* /root/.cache/datalab/models/
    # !cp /kaggle/input/marker-models/GoNotoCurrent-Regular.ttf /usr/local/lib/python3.11/dist-packages/static/fonts/
    import sys
    sys.path.append('/kaggle/input/xml-parse')
    sys.path.append('/kaggle/input/mdc-tools')
    sys.path.append('/kaggle/input/mdc-tools-v2')
from utils import *
from bert_inference import *
from xml_prase import *
# vLLM V1 does not currently accept logits processor so we need to disable it
# https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html#deprecated-features
os.environ["VLLM_USE_V1"] = "0"
if LOCAL:
    labels_dir = 'train_labels.csv'
else:
    labels_dir =  "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"


def load_all_accession_ids_efficient(base_path='/kaggle/input/accession-ids-dataset'):
    """
    高效版本：逐行读取，避免内存溢出。
    同时收集每个 ID 对应的 PMCID 和 EXTID 列表。
    """
    all_ids = set()
    # 新增字典用于存储 ID 对应的 PMCID 和 EXTID
    id_details = {}  # key: ID, value: {'pmc_ids': set(), 'ext_ids': set()}

    csv_files = list(Path(base_path).glob('*.csv'))
    print(f"找到 {len(csv_files)} 个CSV文件")

    for csv_file in csv_files:
        # 假设列名是文件名（不含扩展名）
        column_name = csv_file.stem

        try:
            # 分块读取大文件
            chunk_size = 10000
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                if column_name in chunk.columns:
                    # 处理当前块

                    # 1. 提取主ID（原逻辑）
                    ids_series = chunk[column_name].dropna().astype(str)
                    ids_list = ids_series.tolist()
                    all_ids.update(ids_list)

                    # 2. 提取 PMCID 和 EXTID (如果列存在)
                    pmc_col = 'PMCID'
                    ext_col = 'EXTID'

                    # 确保列存在再处理
                    if pmc_col in chunk.columns and ext_col in chunk.columns:
                        # 遍历当前块的行，填充 id_details 字典
                        for _, row in chunk.iterrows():
                            main_id = row.get(column_name)
                            if pd.isna(main_id):
                                continue  # 跳过主ID为空的行
                            main_id = str(main_id)

                            # 初始化字典条目
                            if main_id not in id_details:
                                id_details[main_id] = {'pmc_ids': set(), 'ext_ids': set()}

                            # 添加 PMCID (如果非空)
                            pmc_id = row.get(pmc_col)
                            if not pd.isna(pmc_id):
                                id_details[main_id]['pmc_ids'].add(str(pmc_id))

                            # 添加 EXT_ID (如果非空)
                            ext_id = row.get(ext_col)
                            if not pd.isna(ext_id):
                                id_details[main_id]['ext_ids'].add(str(ext_id))
                    # 如果列不存在，可以打印警告或忽略
                    # else:
                    #     print(f"  警告: 文件 {csv_file.name} 缺少 '{pmc_col}' 或 '{ext_col}' 列")

            print(f"✓ {column_name}: 处理完成")

        except Exception as e:
            print(f"✗ {csv_file.name}: 读取错误 - {str(e)}")

    print(f"\n总共收集到 {len(all_ids)} 个唯一的accession IDs")
    # 可选：打印 id_details 的大小作为调试信息
    # print(f"收集到 {len(id_details)} 个 ID 的详细信息")

    # 返回集合和字典
    return all_ids, id_details

# 我们将部分具有代表性的上下文规则提取出来，并编译正则表达式以提高效率
# 注意：(?i) 标志在XML中，我们在 re.compile 中使用 re.IGNORECASE 来实现
# import re
bad_ids_acc = {"cath_domain", "pdb_id", "refsnp_id", "ena_accession", "uniprot_id", "cath_id", 'brenda_ec_number'}
CONTEXT_VALIDATION_RULES = {
    'arrayexpress': {
        'context_regex': re.compile(r'(arrayexpress|atlas|gxa|accession|experiment)', re.IGNORECASE),
        'window_size': 5000
    },
    'alphafold': {
        'context_regex': re.compile(
            r'(alphafold|alphafold database|alphafold db|structures|predicted structure|predicted protein structure|protein|identifier|accession)',
            re.IGNORECASE),
        'window_size': 5000
    },
    'brenda': {
        'context_regex': re.compile(
            r'(BRENDA enzyme|BRENDA enzyme database|BRaunschweig ENzyme DAtabase|enzyme database|enzyme|lysosomes|lysosomal|BRENDA:|BRENDA: |BRENDA:EC|BRENDA:EC|BRENDA tissue ontology|BTO|ontology)',
            re.IGNORECASE),
        'window_size': 2000
    },
    'bia': {
        'context_regex': re.compile(r'(bia|bioimage archive database|bioimage archive|database|identifier|accession)',
                                    re.IGNORECASE),
        'window_size': 5000
    },
    'biomodels': {
        'context_regex': re.compile(r'(biomodels|accession|model|identifier)', re.IGNORECASE),
        'window_size': 3000
    },
    'biosample': {
        'context_regex': re.compile(r'(biosample|accession|model)', re.IGNORECASE),
        'window_size': 5000
    },
    'cath': {
        'context_regex': re.compile(
            r'(cath|cath-Gene3D|cath Gene3D|c\.a\.t\.h|domain|families|cathnode|pdb|superfamily)', re.IGNORECASE),
        'window_size': 5000
    },
    'cellosaurus': {
        'context_regex': re.compile(
            r'(cells|cellosaurus|cellosaurus database|Cell lines|Cell Bank|cell lines|cell bank|accession number|RRID:)',
            re.IGNORECASE),
        'window_size': 5000
    },
    'chebi': {
        'context_regex': re.compile(r'(chebi|compound)', re.IGNORECASE),
        'window_size': 5000
    },
    'chembl': {
        'context_regex': re.compile(r'(chembl|compound)', re.IGNORECASE),
        'window_size': 5000
    },
    'complexportal': {
        'context_regex': re.compile(r'(protein|complex)', re.IGNORECASE),
        'window_size': 5000
    },
    'metagenomics': {
        'context_regex': re.compile(r'(samples|ebi metagenomics|metagenomics|database)', re.IGNORECASE),
        'window_size': 5000
    },
    'ega': {  # Combined from ega.study, ega.dataset, ega.dac
        'context_regex': re.compile(
            r'(ega|accession|archive|studies|study|dataset|datasets|data set|data sets|validation sets|validation set|set|sets|data|dac|European Genome-phenome Archive|European Genome phenome Archive)',
            re.IGNORECASE),
        'window_size': 5000
    },
    'emdb': {
        'context_regex': re.compile(r'(emdb|accession|code)', re.IGNORECASE),
        'window_size': 5000
    },
    'ena': {  # Combined from multiple gen/ena rules
        'context_regex': re.compile(
            r'(genbank|\bgen\b|\bena\b|ddbj|embl|european nucleotide archive|accession|nucleotide|archive|asssembled|annotated|sequence|sequences|protein coding|protein|trace|traces|study|studies|sample|samples|experiment|experiments|run|runs|analysis|analyses|submission|submissions)',
            re.IGNORECASE),
        'window_size': 2000
    },
    'ensembl': {
        'context_regex': re.compile(r'(ensembl|accession|transcript|sequence)', re.IGNORECASE),
        'window_size': 5000
    },
    'go': {
        'context_regex': re.compile(r'(go|gene ontology)', re.IGNORECASE),
        'window_size': 5000
    },
    'hgnc': {
        'context_regex': re.compile(r'(HUGO Gene Nomenclature Committee|hugo|gene|nomenclature|committee|database)',
                                    re.IGNORECASE),
        'window_size': 5000
    },
    'igsr': {
        'context_regex': re.compile(
            r'(\bcell\b|sample|iPSC|iPSCs|iPS|fibroblast|fibroblasts|QTL|eQTL|pluripotent|induced|\bdonor\b|\bstem\b|EBiSC|1000 Genomes|Coriell|\bLCL\b|lymphoblastoid)',
            re.IGNORECASE),
        'window_size': 3000
    },
    'intact': {
        'context_regex': re.compile(r'(intact|IntAct|inTact|Intact|interaction|interactions|protein)', re.IGNORECASE),
        'window_size': 5000
    },
    'mint': {
        'context_regex': re.compile(r'(MINT|molecular interaction database|interactions|interaction)', re.IGNORECASE),
        'window_size': 5000
    },
    'interpro': {
        'context_regex': re.compile(r'(interpro|domain|family|motif|accession)', re.IGNORECASE),
        'window_size': 5000
    },
    'metabolights': {
        'context_regex': re.compile(r'(metabolights|accession|repository)', re.IGNORECASE),
        'window_size': 5000
    },
    'pdb': {
        'context_regex': re.compile(r'(pdb|(?:protein\s+data\s*bank)|accession|structure|domain)', re.IGNORECASE),
        'window_size': 1000
    },
    'pfam': {
        'context_regex': re.compile(r'(pfam|domain|family|accession|motif)', re.IGNORECASE),
        'window_size': 5000
    },
    'pxd': {
        'context_regex': re.compile(r'(pxd|proteomexchange|pride|dataset|accession|repository)', re.IGNORECASE),
        'window_size': 5000
    },
    'reactome': {
        'context_regex': re.compile(r'(biological|regulatory|pathway|pathways|database)', re.IGNORECASE),
        'window_size': 5000
    },
    'rhea': {
        'context_regex': re.compile(r'(reactions|database|rhea database|accession)', re.IGNORECASE),
        'window_size': 5000
    },
    'uniprot': {
        'context_regex': re.compile(
            r'(swiss-prot|sprot|uniprot|swiss prot|accession(s)?|Locus|GenBank|genome|sequence(s)?|protein|trembl|uniparc|uniprotkb|Acc\.No|Acc\. No)',
            re.IGNORECASE),
        'window_size': 5000
    },
    'uniparc': {
        'context_regex': re.compile(r'(uniprot|accession(s)?|Locus|sequence(s)?|protein|uniparc|Acc\.No|Acc\. No)',
                                    re.IGNORECASE),
        'window_size': 5000
    },
    'ebisc': {
        'context_regex': re.compile(
            r'(\bcell\b|sample|iPSC|iPSCs|iPS|fibroblast|fibroblasts|QTL|eQTL|pluripotent|induced|\bdonor\b|\bstem\b|EBiSC|1000 Genomes|Coriell|\bLCL\b|lymphoblastoid)'),
        'window_size': 5000
    },
    'hipsci': {
        'context_regex': re.compile(
            r'(\bcell\b|sample|iPSC|iPSCs|iPS|fibroblast|fibroblasts|QTL|eQTL|pluripotent|induced|\bdonor\b|\bstem\b|EBiSC|1000 Genomes|Coriell|\bLCL\b|lymphoblastoid)'),
        'window_size': 5000
    },
    'refseq': {
        'context_regex': re.compile(r'(refseq|genbank|accession|sequence)', re.IGNORECASE),
        'window_size': 5000
    },
    'refsnp': {
        'context_regex': re.compile(
            r'(allele|model|multivariate|polymorphism|locus|loci|haplotype|genotype|variant|chromosome|SNPs|snp|snp(s)*)',
            re.IGNORECASE),
        'window_size': 5000
    },
    'doi': {
        'context_regex': re.compile(r'(doi|repository)', re.IGNORECASE),
        'window_size': 5000
    },
    'bioproject': {
        'context_regex': re.compile(r'(bioproject|accession|archive)', re.IGNORECASE),
        'window_size': 5000
    },
    'treefam': {
        'context_regex': re.compile(r'(treefam|tree|family|accession|dendrogram)', re.IGNORECASE),
        'window_size': 5000
    },
    'eudract': {
        'context_regex': re.compile(r'(eudract|trial|agency|register|clinical)', re.IGNORECASE),
        'window_size': 5000
    },
    'nct': {
        'context_regex': re.compile(r'(trial)', re.IGNORECASE),
        'window_size': 5000
    },
    'dbgap': {
        'context_regex': re.compile(
            r'(database of genotypes and phenotypes|dbgap|accession|archives|studies|interaction)', re.IGNORECASE),
        'window_size': 5000
    },
    'geo': {
        'context_regex': re.compile(
            r'(gene expression omnibus|genome|geo|accession|functional genomics|data repository|data submissions)',
            re.IGNORECASE),
        'window_size': 5000
    },
    'orphadata': {
        'context_regex': re.compile(
            r'(database|rare disease|disease|data|nomenclature|syndrome|id|number|name|orphanet|orphadata|orpha)',
            re.IGNORECASE),
        'window_size': 5000
    },
    'gisaid': {
        'context_regex': re.compile(
            r'(gisaid|global initiative on sharing all influenza data|segment|segments|identifier|flu|epi|epiflu|database|sequence|sequences|isolate|isolates|accession|virus|viruses|strain|strains)',
            re.IGNORECASE),
        'window_size': 5000
    },
}
SPECIAL_MAPPINGS = {
    'sra_id': 'ena',
    'ena_trace': 'ena',
    'pride_id': 'pxd',
}


def is_context_valid(text: str, context_regex: re.Pattern) -> bool:
    """
    检查一个正则表达式匹配项（match object）的上下文是否有效。

    Args:
        match: re.finditer 返回的 match 对象。
        text: 供搜索的完整原始文本。
        context_regex: 用于验证上下文的已编译的正则表达式。
        window_size: 在匹配项前后搜索上下文关键字的字符数。

    Returns:
        如果上下文有效，则返回 True，否则返回 False。
    """
    # 获取匹配项在文本中的起始和结束位置
    match_start, match_end = match.span()

    # 根据 window_size 计算上下文搜索窗口的边界
    # 确保窗口边界不会超出文本的实际长度
    window_start = max(0, match_start - window_size)
    window_end = min(len(text), match_end + window_size)

    # 提取上下文窗口的文本
    context_window_text = text[window_start:window_end]

    # 在上下文窗口中搜索关键字
    if context_regex.search(text):
        return True  # 找到了上下文关键字，验证通过
    else:
        return False  # 未找到上下文关键字，验证失败

CLASSIFICATION = {
    "primary": re.compile(r'(deposited|accessible|submitted|measured|collected|observed|detected|analyzed|generated|produced|obtained|recorded|sequenced|amplified|isolated|vailability|vailable|novel|hosted|raw(?:\s+data)?)|supplementary'),
}


def find_and_cluster_keywords(body_text, rules, window_size=100, cluster_gap=50):
    """
    在文本中查找关键词，将邻近的匹配项聚类，并为每个聚类提取唯一的上下文窗口。
    """
    all_matches = []
    for prefix, keyword_reg in rules.items():
        for match in keyword_reg.finditer(body_text):
            all_matches.append({'prefix': prefix, 'match': match})

    if not all_matches:
        return {}

    all_matches.sort(key=lambda x: x['match'].start())

    clusters = []
    if all_matches:
        current_cluster = [all_matches[0]]
        clusters.append(current_cluster)
        for i in range(1, len(all_matches)):
            current_item = all_matches[i]
            last_item_in_cluster = current_cluster[-1]
            gap = current_item['match'].start() - last_item_in_cluster['match'].end()
            if gap < cluster_gap:
                current_cluster.append(current_item)
            else:
                current_cluster = [current_item]
                clusters.append(current_cluster)

    final_results = {}
    for cluster in clusters:
        main_prefix = cluster[0]['prefix']
        cluster_start = cluster[0]['match'].start()
        cluster_end = cluster[-1]['match'].end()
        window_start = max(0, cluster_start - window_size)
        window_end = min(len(body_text), cluster_end + window_size)
        context_window_text = body_text[window_start:window_end].strip()

        context_list = final_results.setdefault(main_prefix, [])
        if context_window_text not in context_list:
            context_list.append(context_window_text)

    return final_results


DOI_KEYWORD_RULES = {
    # Key: DOI Prefix (string)
    # Value: Compiled regex of associated keywords

    "10.15468": re.compile(r"(GBIF|Global Biodiversity Information Facility)"),
    "10.5061": re.compile(r"(Dryad|Dryad Digital Repository)"),
    "10.5281": re.compile(r"(Zenodo|CERN)"),
    "10.6073": re.compile(r"(PASTA|EDI|Environmental Data Initiative)"),
    "10.17605": re.compile(r"(OSF|Open Science Framework)"),
    "10.5256": re.compile(r"(F1000Research)"),
    "10.6084": re.compile(r"(Figshare)"),
    "10.1594": re.compile(r"(Pangaea|Data Publisher for Earth & Environmental Science)"),
    "10.7910": re.compile(r"(Dataverse|Harvard Dataverse)"),
    "10.3886": re.compile(r"(ICPSR|Inter-university Consortium for Political and Social Research)")
}
import re

# (最终版本) 完整覆盖所有参考规则，并遵循“最长优先”原则的字典
ACCESSION_ID_KEYWORD_RULES = {
    # --- 序列、基因组与变异数据库 ---
    "ena": re.compile(r"(European\sNucleotide\sArchive|GenBank|DDBJ|ENA|EMBL)", re.IGNORECASE),
    "refseq": re.compile(r"(Reference\sSequence|RefSeq)", re.IGNORECASE),
    "ensembl": re.compile(r"(Ensembl)", re.IGNORECASE),
    "refsnp": re.compile(r"(dbSNP|rsID|SNP)", re.IGNORECASE),
    "ega": re.compile(r"(European\sGenome-phenome\sArchive|EGA)", re.IGNORECASE),
    "dbgap": re.compile(r"(database\sof\sgenotypes\sand\sphenotypes|dbgap)", re.IGNORECASE),
    "gisaid": re.compile(r"(gisaid|epiflu|EPI_ISL)", re.IGNORECASE),
    "metagenomics": re.compile(r"(ebi\smetagenomics)", re.IGNORECASE),
    "hgnc": re.compile(r"(HUGO\sGene\sNomenclature\sCommittee|hgnc)", re.IGNORECASE),
    "treefam": re.compile(r"(treefam)", re.IGNORECASE),

    # --- 蛋白质与结构数据库 ---
    "uniprot": re.compile(r"(swiss-prot|swiss\sprot|UniProtKB|TrEMBL|UniProt|Sprot)", re.IGNORECASE),
    "uniparc": re.compile(r"(uniparc)", re.IGNORECASE),
    "pdb": re.compile(r"(Protein\sData\sBank|PDB)", re.IGNORECASE),
    "alphafold": re.compile(r"(AlphaFold\sDB|AlphaFold)", re.IGNORECASE),
    "pfam": re.compile(r"(Pfam)", re.IGNORECASE),
    "interpro": re.compile(r"(InterPro)", re.IGNORECASE),
    "cath": re.compile(r"(cath-Gene3D|cath)", re.IGNORECASE),
    "emdb": re.compile(r"(emdb)", re.IGNORECASE),

    # --- 功能基因组学与表达数据库 ---
    "geo": re.compile(r"(Gene\sExpression\sOmnibus|GEO)", re.IGNORECASE),
    "arrayexpress": re.compile(r"(ArrayExpress|gxa|atlas)", re.IGNORECASE),

    # --- 化学、代谢物与反应数据库 ---
    "chebi": re.compile(r"(ChEBI)", re.IGNORECASE),
    "chembl": re.compile(r"(ChEMBL)", re.IGNORECASE),
    "brenda": re.compile(r"(BRaunschweig\sENzyme\sDAtabase|BRENDA\senzyme|BRENDA|BTO)", re.IGNORECASE),
    "rhea": re.compile(r"(rhea)", re.IGNORECASE),
    "metabolights": re.compile(r"(metabolights)", re.IGNORECASE),

    # --- 系统生物学与通路数据库 ---
    "biomodels": re.compile(r"(BioModels)", re.IGNORECASE),
    "reactome": re.compile(r"(Reactome)", re.IGNORECASE),
    "intact": re.compile(r"(IntAct)", re.IGNORECASE),
    "mint": re.compile(r"(molecular\sinteraction\sdatabase|MINT)", re.IGNORECASE),
    "complexportal": re.compile(r"(ComplexPortal)", re.IGNORECASE),

    # --- 样本、细胞系与项目数据库 ---
    "bioproject": re.compile(r"(BioProject|NCBI\sBioProject)", re.IGNORECASE),
    "biosample": re.compile(r"(BioSample|NCBI\sBioSample)", re.IGNORECASE),
    "cellosaurus": re.compile(r"(Cellosaurus|CVCL)", re.IGNORECASE),
    "igsr": re.compile(r"(1000\sGenomes|Coriell|igsr)", re.IGNORECASE),
    "ebisc": re.compile(r"(EBiSC)", re.IGNORECASE),
    "hipsci": re.compile(r"(HipSci)", re.IGNORECASE),
    "pxd": re.compile(r"(ProteomeXchange|PRIDE|PXD)", re.IGNORECASE),
    "bia": re.compile(r"(bioimage\sarchive|bia)", re.IGNORECASE),

    # --- 本体论与疾病数据库 ---
    "go": re.compile(r"(Gene\sOntology|GO)", re.IGNORECASE),
    "orphadata": re.compile(r"(Orphanet|Orphadata|ORPHA)", re.IGNORECASE),

    # --- 临床试验数据库 ---
    "nct": re.compile(r"(ClinicalTrials.gov|NCT)", re.IGNORECASE),
    "eudract": re.compile(r"(EudraCT)", re.IGNORECASE),
}


def fix_broken_links(text: str) -> str:
    patterns = [
        (r'https?://\s*([a-zA-Z0-9\.\-]+)', r'https://\1'),
        (r'(doi\.)\s+(org/10\.)', r'\1\2'),
        (r'doi\.org/\s+10\.', r'doi.org/10.'),
        (r'doi\.org/10\.\s*(\d+)', r'doi.org/10.\1'),
        (r'doi\.org/10\.(\d{4,5})/\s+([a-zA-Z0-9\.\-_]+)', r'doi.org/10.\1/\2'),
        # (r'(https?://[a-zA-Z0-9\.\-/]+/)\s+([a-zA-Z0-9\.\-_/]+)', r'\1\2'),
        # (r'(https?://[a-zA-Z0-9\.\-/]+-)\s+([a-zA-Z0-9\.\-_/]+)', r'\1\2'),
        (
        r'(10\.\d{4,5}/[a-zA-Z0-9\.\-/]+\.)\s+([a-z0-9\.\-_/]{4,20},?\s+|(?!(?:DATAS?|RESULTS?|DISCUSSIONS?|REFERENCES?|CONFLICT|ORCID|ACKNOWLEDGMENTS?|References?)\b)[A-Z0-9\.\-_/]{4,20}\s+)',
        r'\1\2'),
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

def add_period_to_heading_lines(text):
    """
    在所有匹配BIBLIOGRAPHY_PATTERNS的标题行末尾添加句点
    条件：行必须完全匹配预定义模式且不以标点结尾
    """
    lines = text.split('\n')

    for i in range(len(lines)):
        line = lines[i].rstrip()  # 只移除右侧空白

        # 检查是否匹配任何参考文献/标题模式
        if any(re.fullmatch(pattern, line, re.IGNORECASE) for pattern in BIBLIOGRAPHY_PATTERNS):
            # 检查行尾是否已有标点
            # print("有目标",line)
            if not re.search(r'[.]$', line):
                lines[i] = '\n\n' + line + '\n\n'
                # print(f"添加逗号: {line}")

    return '\n'.join(lines)


def normalize_text(text: str) -> str:
    """
    对单个字符串进行标准化处理：
    1. Unicode 标准化（NFKC）
    2. 移除非 ASCII 字符
    3. 将 Zenodo URL 转换为 DOI 格式
    """
    # 1. Unicode 标准化
    normalized = unicodedata.normalize("NFKC", text)
    # 2. 移除非 ASCII 字符
    ascii_only = re.sub(r"[^\x00-\x7F]", "", normalized)
    # 3. Zenodo URL 转 DOI
    zenodo_doi = re.sub(
        r"https?://zenodo\.org/record/(\d+)",
        r" 10.5281/zenodo.\1 ",
        ascii_only
    )
    return zenodo_doi.strip()  # 移除多余空格


def find_last_reference_header(text: str, header_patterns: list[re.Pattern]) -> Optional[int]:
    last_match_idx = None
    for pattern in header_patterns:
        matches = list(pattern.finditer(text))
        if matches:
            last_match_idx = matches[-1].start()
    return last_match_idx


def find_last_first_citation(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_match_line = None
    for line_num, line in enumerate(lines):
        line = line.strip()
        for pattern in COMPILED_PATTERNS['first_citation_patterns']:
            if pattern.match(line):
                next_lines = lines[line_num:line_num + 3]
                if any(COMPILED_PATTERNS['citation_pattern'].match(l.strip()) for l in next_lines[1:]):
                    last_match_line = line_num
                break
    return last_match_line


def find_reference_start(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_first_citation = find_last_first_citation(text)
    if last_first_citation is not None:
        return last_first_citation
    start_search_idx = int(len(lines) * 0.5)
    for i in range(start_search_idx, len(lines)):
        line = lines[i].strip()
        if COMPILED_PATTERNS['citation_pattern'].match(line):
            next_lines = lines[i:i + 3]
            if sum(1 for l in next_lines if COMPILED_PATTERNS['citation_pattern'].match(l.strip())) >= 2:
                for j in range(i, max(-1, i - 10), -1):
                    if not COMPILED_PATTERNS['citation_pattern'].match(lines[j].strip()):
                        return j + 1
                return max(0, i - 10)
    return None


def split_text_and_references(text: str) -> Tuple[str, str]:
    header_idx = find_last_reference_header(text, COMPILED_PATTERNS['ref_header_patterns'])
    if header_idx is not None:
        header_idx2 = find_last_reference_header(text[:header_idx].strip(), COMPILED_PATTERNS['ref_header_patterns'])
        if header_idx2 is not None:
            header_idx3 = find_last_reference_header(text[:header_idx2].strip(),
                                                     COMPILED_PATTERNS['ref_header_patterns'])
            if header_idx3 is not None:
                return text[:header_idx3].strip(), text[header_idx3:].strip()
            return text[:header_idx2].strip(), text[header_idx2:].strip()
        return text[:header_idx].strip(), text[header_idx:].strip()
    ref_start_line = find_reference_start(text)
    if ref_start_line is not None:
        lines = text.splitlines()
        body = '\n'.join(lines[:ref_start_line])
        refs = '\n'.join(lines[ref_start_line:])
        return body.strip(), refs.strip()
    return text.strip(), ''

def is_balanced(text: str) -> bool:
    """检查字符串中的括号是否平衡"""
    paren_stack = 0  # 圆括号计数器
    bracket_stack = 0  # 方括号计数器

    for char in text:
        if char == '(':
            paren_stack += 1
        elif char == ')':
            paren_stack -= 1
            if paren_stack < 0:  # 出现多余的闭括号
                return False
        elif char == '[':
            bracket_stack += 1
        elif char == ']':
            bracket_stack -= 1
            if bracket_stack < 0:
                return False

    return paren_stack == 0 and bracket_stack == 0  # 所有括号必须完全闭合

def remove_references_section_v2(text):
    lines = text.split('\n')
    cut_index = -1

    # Look backwards from end of document
    for i in range(len(lines) - 1, max(0, int(len(lines) * 0.3)), -1):
        line = lines[i].strip()

        if any(re.match(pattern, line, re.IGNORECASE) for pattern in BIBLIOGRAPHY_PATTERNS):
            # Double-check: look at following lines for citation patterns
            following_lines = lines[i + 1:i + 5]  # Check more lines
            has_citations = False

            for follow_line in following_lines:
                if follow_line.strip():
                    # Check for obvious citation patterns
                    if (re.search(r'\(\d{4}\)', follow_line) or
                            re.search(r'\d{4}\.', follow_line) or
                            'doi:' in follow_line.lower() or
                            ' et al' in follow_line.lower() or
                            re.search(r'^\[\d+\]', follow_line.strip()) or  # [1], [2], etc.
                            re.search(r'^\d+\.', follow_line.strip())):  # 1., 2., etc.
                        has_citations = True
                        break

            # Only cut if we found citation-like content
            if has_citations or i >= len(lines) - 3:  # Or very near end
                cut_index = i
                break

    if cut_index != -1:
        return '\n'.join(lines[:cut_index]).strip(), '\n'.join(lines[cut_index:]).strip()

    return text.strip(), ""

def remove_references_section_v3(text):
    lines = text.split('\n')
    cut_index = -1

    # 从文本的后70%部分开始，从后向前搜索
    # 这是一个合理的启发式搜索，避免在长文档的开头部分进行不必要的工作
    start_search_index = int(len(lines) * 0.3)
    for i in range(len(lines) - 1, start_search_index, -1):
        line = lines[i].strip()

        # 检查当前行是否与某个标题模式完全匹配
        if any(re.fullmatch(pattern, line, re.IGNORECASE) for pattern in BIBLIOGRAPHY_PATTERNS):

            # --- 开始进行更严格的“双重验证” ---

            # 验证 1: 检查该行是否符合典型的“标题格式”
            is_heading_format = (
                    len(line) < 40 and  # 标题长度通常不会太长
                    (line.isupper() or line.istitle())  # 典型格式：全大写或首字母大写
            )

            # 验证 2: 统计后续几行中的“引文特征”数量
            citation_features_count = 0
            following_lines_to_check = lines[i + 1: i + 10]  # 向后看7行进行验证

            for follow_line in following_lines_to_check:
                if (re.search(r'\((19|20)\d{2}\)', follow_line) or  # (Author, 2020)
                        re.search(r'\b(19|20)\d{2}[a-z]?\b', follow_line) or  # 2020 or 2020a
                        'doi:' in follow_line.lower() or
                        'et al' in follow_line.lower() or
                        re.search(r'^\[\d+\]', follow_line.strip()) or  # [1]
                        re.search(r'^\d+\.\s', follow_line.strip()) or
                        re.search(r'[A-Z]\.', follow_line.strip())):  # 1. Author
                    citation_features_count += 1

            # --- 决策逻辑 ---
            # 如果它看起来像一个标题，或者其后紧跟着高密度的引文内容，
            # 我们就认为找到了切分点。
            # (阈值设为2，表示后续7行中至少有2行包含引文特征，这是一个强信号)
            if is_heading_format and citation_features_count >= 5:
                cut_index = i
                # break  # 找到后立即停止向前的搜索
    # 如果主搜索未找到，则启用备用搜索策略
    if cut_index == -1:
        # 在最后若干行中尝试识别高密度引文段
        window_size = 10
        min_citation_density = 9

        # 从后往前滑动窗口检查
        for i in range(len(lines) - window_size, start_search_index, -1):
            window_lines = lines[i:i + window_size]
            citation_features_count = 0

            for line in window_lines:
                stripped = line.strip()
                if (re.search(r'\((19|20)\d{2}\)', stripped) or
                        re.search(r'\b(19|20)\d{2}[a-z]?\b', stripped) or
                        'doi:' in stripped.lower() or
                        'et al' in stripped.lower() or
                        re.search(r'^\[\d+\]', stripped) or
                        re.search(r'^\d+\.\s', stripped) or
                        re.search(r'[A-Z]\.', stripped)):
                    citation_features_count += 1

            if citation_features_count >= min_citation_density:
                cut_index = i
    if cut_index != -1:
        # 如果找到了切分点，返回它之前的所有内容
        return '\n'.join(lines[:cut_index]).strip(), '\n'.join(lines[cut_index:]).strip()

    # 如果没有找到，返回原始文本（去除首尾空白）
    return text.strip(), ""


def find_data_availability_statement(content: str, content_type: str = 'text') -> tuple[str | None, str]:
    """
    使用分层规则从学术文本中查找数据可用性声明 (DAS)。

    Args:
        content (str): 文献的完整内容，可以是纯文本或JATS XML格式的字符串。
        content_type (str): 内容类型，'text' 或 'xml'。

    Returns:
        tuple[str | None, str]: 一个元组，包含找到的声明文本（或None）和使用的方法。
    """
    # # --- 方法一：解析XML标签 (最高优先级) ---
    # if content_type.lower() == 'xml':
    #     try:
    #         statement, method = _find_by_xml_tag(content)
    #         if statement:
    #             return statement, method
    #         # 如果XML中没找到特定标签，将其转换为纯文本以进行后续方法
    #         soup = BeautifulSoup(content, 'lxml-xml')
    #         plain_text = soup.get_text(separator='\n\n', strip=True)
    #     except Exception:
    #         # 如果XML解析失败，则假定内容为纯文本
    #         plain_text = content
    # else:
    #     plain_text = content

    # --- 方法二：通过章节标题查找 (第二优先级) ---
    statement_head, method_header = _find_by_section_header(content)

    # --- 方法三：通过关键词和标识符搜索 (第三优先级) ---
    statement_search, method_search = _find_by_keyword_search(content)

    if statement_head and statement_search:
        return statement_search + '\n' + statement_head, "hybird"
    elif statement_head:
        return statement_head, method_header
    elif statement_search:
        return statement_search, method_search

    return None, "Not Found"


def _find_by_section_header(plain_text: str) -> tuple[str | None, str]:
    """
    通过正则表达式查找标准的DAS章节标题，并提取其后的固定长度文本。
    此版本经过优化，可以处理标题与正文在同一行的情况。
    """
    # 优化的正则表达式:
    # 1. (?:^|\n\n): 匹配文本开头或一个新段落的开头，确保我们找到的是一个真正的标题。
    # 2. \s*: 匹配标题前的任何空白符。
    # 3. ((data|...|le)|data\s*deposition): 捕获核心的标题关键词。
    # 4. [\s.:;]*: 匹配标题后可能出现的空格、句号、冒号、分号，使其更灵活。
    header_regex = re.compile(
        r"((?:data|code|software|materials)\s*(?:and\s+materials\s*)?availab(?:ility|le)|data\s*deposition)[\s.:;]*",
        re.IGNORECASE
    )

    # 在整个文本中搜索第一次出现的标题
    match = header_regex.search(plain_text)

    if match:
        # 找到匹配项的起始位置
        start_index = match.start()

        # 从标题开始的位置，向后提取最多500个字符
        # 使用 strip() 去除开头可能存在的换行符和空格
        end_index = start_index + 500
        full_statement = plain_text[start_index:end_index].strip()

        return full_statement, "Section Header"

    return None, "Section Header"

def _find_by_keyword_search(plain_text: str) -> Tuple[Optional[str], str]:
    """在全文中搜索包含数据来源描述的句子，并应用长度和距离限制。"""

    # 数据来源关键词（包括 submitted, obtained from, GISAID 等）
    # 数据来源关键词（“动作”或“通用概念”）- 精简版
    data_source_keywords = re.compile(
        r'\b('
        # --- 强一手信号：数据创建与章节标题 ---
        r'data\s+and\s+materials\s+availability|'
        r'data\s+availability|code\s+availability|'
        r'underlying\s+data|source\s+data|source\s+code|'
        r'generated|produced|collected|'

        # --- 常见一手/中性信号：提交与发布 ---
        r'submitted|deposited|released|archived|'
        r'data\s+release|'

        # --- 中性可用性描述 ---
        r'provided\s+by|available\s+at|available\s+in|available\s+from|'
        r'can\s+be\s+found|are\s+provided|are\s+included|hosted|'

        # --- 中性核心名词 ---
        r'dataset|datasets|data\s+set|accession|availability'
        r')\b',
        re.IGNORECASE
    )

    # 数据存储库/标识符（用于确认是真实的数据来源）
    repo_or_id_pattern = re.compile(
        r'('
        # --- Major Repository Names ---
        r'\b(zenodo|figshare|dryad|github|gitlab|GBIF)\b|'
        r'\b(AlphaFold\sDB|AlphaFold|ArrayExpress|'
        r'BioImage\sArchive|bia|BioModels|NCBI\sBioProject|BioProject|NCBI\sBioSample|BioSample|'
        r'BRaunschweig\sENzyme\sDAtabase|BRENDA\senzyme|BRENDA|BTO|'
        r'cath-Gene3D|cath|Cellosaurus|ChEBI|ChEMBL|ClinicalTrials\.gov|'
        r'ComplexPortal|Coriell|'
        r'database\sof\sgenotypes\sand\sphenotypes|dbgap|dbSNP|'
        r'ebi\smetagenomics|EBiSC|EMBL|EMDB|EMPIAR|Ensembl|'
        r'European\sGenome-phenome\sArchive|EGA|'
        r'European\sNucleotide\sArchive|EudraCT|'
        r'Gene\sExpression\sOmnibus|GEO|Gene\sOntology|GenBank|gisaid|gxa|'
        r'HipSci|HUGO\sGene\sNomenclature\sCommittee|hgnc|'
        r'IntAct|InterPro|'
        r'MetaboLights|molecular\sinteraction\sdatabase|MINT|'
        r'Orphanet|Orphadata|ORPHA|'
        r'Pangaea|Pfam|Protein\sData\sBank|PDB|ProteomeXchange|PRIDE|'
        r'Reactome|Reference\sSequence|RefSeq|rhea|RNAcentral|'
        r'swiss-prot|swiss\sprot|UniProtKB|TrEMBL|UniProt|Sprot|uniparc|'
        r'TreeFam|1000\sGenomes)\b'
        r')',
        re.IGNORECASE
    )

    # 句子分割（改进版，处理缩写和复杂标点）
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', plain_text)

    matched_sentences = []

    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean:
            continue

        # 规则1：句子必须包含两个关键词
        # **改动1：获取匹配对象本身，而不是布尔值，以便后续计算位置**
        source_match = data_source_keywords.search(sent_clean)
        repo_match = repo_or_id_pattern.search(sent_clean)

        if source_match and repo_match:
            # --- 新增逻辑：开始 ---

            # **规则2：检查两个关键词之间的距离**
            # 计算两个匹配项之间的距离（后一个的开头 - 前一个的结尾）
            first_end = min(source_match.end(), repo_match.end())
            second_start = max(source_match.start(), repo_match.start())

            # 修正：应该是后一个的开头 - 前一个的结尾
            if source_match.start() < repo_match.start():
                # source_match 在前
                distance = repo_match.start() - source_match.end()
            else:
                # repo_match 在前
                distance = source_match.start() - repo_match.end()

            # 如果距离超过200个字符，则跳过这个句子
            if distance > 200:
                continue

            # **规则3：如果句子太长，则进行精简**
            final_snippet = sent_clean
            if len(sent_clean) > 300:
                # 确定两个关键词共同覆盖的核心区域
                interest_start = min(source_match.start(), repo_match.start())
                interest_end = max(source_match.end(), repo_match.end())

                # 计算核心区域的中点
                midpoint = interest_start + (interest_end - interest_start) // 2

                # 以中点为中心，向两边扩展，总长度为200
                snippet_start = max(0, midpoint - 150)
                snippet_end = min(len(sent_clean), snippet_start + 300)

                # 确保窗口调整后不会切断开头的单词
                if snippet_start > 0:
                    snippet_start = sent_clean.find(' ', snippet_start) + 1

                snippet = sent_clean[snippet_start:snippet_end]

                # 如果截取了开头或结尾，则加上省略号
                prefix = "... " if snippet_start > 0 else ""
                suffix = " ..." if snippet_end < len(sent_clean) else ""

                final_snippet = f"{prefix}{snippet}{suffix}"

            # --- 新增逻辑：结束 ---

            matched_sentences.append(final_snippet)

    if matched_sentences:
        return "\n".join(matched_sentences), "Data Source Detection"

    return None, "No data source mentions found"

df_doi = pd.read_csv('/kaggle/input/d/patricklo01/accession-ids-dataset/doi.csv')
doi_series = df_doi["doi"].dropna().astype(str)
prefixes = doi_series.str.split('/', n=1).str[0]
doi_prefix_set = set(prefixes)

unique_datasets = "/kaggle/input/d/patricklo01/data-doi-and-accesion-ids/unique_datasets.txt"
reorganized_publication_dataset = "/kaggle/input/data-reorganized-publication-dataset/reorganized_publication_dataset.json"
chunks = []
chunks_ref = []
cover_information = []
article_data_information = {}
database_information = {}
text_span_len = 100
chunk_size = 200
overlap = 50
found_items = []
j = 0
bad_ids = ["10.5061/dryad", "10.6073/pasta", "10.5281/zenodo", "10.25386/genetics", "10.1175/jcli"]
journal_ids = [
    "10.18637/jss", "journal", "10.1109/cvpr", "figshare", "10.1111", "10.1016/j.",
    "10.1002/", "10.1038/", "10.1126/", "10.1073/", "10.1093/", "10.1103/",
    "10.1021/", "10.1007/", "10.1186/", "10.1371/", "10.3390/", "10.1155/",
    "10.1159/", "10.1161/", "10.1210/", "10.1212/", "10.1214/", "10.1215/",
    "10.1242/", "10.1256/", "10.1261/", "10.1289/", "10.1299/", "10.1300/",
    "10.1310/", "10.1320/", "10.1330/", "10.1340/", "10.1350/", "10.1360/",
    "10.1370/", "10.1380/", "10.1390/", "10.1400/", "10.1410/", "10.1420/",
    "10.1430/", "10.1440/", "10.1450/", "10.1460/", "10.1470/", "10.1480/",
    "10.1490/", "10.1500/", "10.1510/", "10.1520/", "10.1530/", "10.1540/",
    "10.1550/", "10.1560/", "10.1570/", "10.1580/", "10.1590/", "10.1600/",
    "10.1610/", "10.1620/", "10.1630/", "10.1640/", "10.1650/", "10.1660/",
    "10.1670/", "10.1680/", "10.1690/", "10.1700/", "10.1710/", "10.1720/",
    "10.1730/", "10.1740/", "10.1750/", "10.1760/", "10.1770/", "10.1780/",
    "10.1790/", "10.1800/", "10.1810/", "10.1820/", "10.1830/", "10.1840/",
    "10.1850/", "10.1860/", "10.1870/", "10.1880/", "10.1890/", "10.1900/",
    "10.1910/", "10.1920/", "10.1930/", "10.1940/", "10.1950/", "10.1960/",
    "10.1970/", "10.1980/", "10.1990/", "10.1080/", "10.1029/", "10.1143/",
    "10.3238/arztebl", "10.3322/caac", "10.2458/azu_js_rc", "10.21105/joss", "10.1089",
    "10.5194/gmd", "10.1146/annurev", "10.1098/rspb", "10.3897/neobiota", "10.3763/ghgmm",
    "10.1175/jhm", "10.5194/acp", "10.5194/hess", "10.1098/", "10.11613/bm", "10.4161/rna",
    "10.2307", "10.3354", "10.4159/harvard", "10.1051/forest", "10.1098/rstb", "10.1127",
    "10.1086/", "10.1071/", "10.2193/", "10.1641/", "10.3201/", "10.1162/", "10.14806/ej",
    "10.1669/", "10.1006/", "10.1152", "10.2525", "10.2217/", "10.14411", "10.14440/",
    "10.5665/", "10.1113/", "10.1586/", "10.1669/", "10.1177/", "10.1895/", "10.13039/",
    "10.1128/", "10.1191/", "10.1109", "10.4049/", "10.1145", "10.3389/fimmu", "10.1175/",
    "10.2202/", "10.1175/", "10.4061/",
]


def journal_is_in_match(match: str) -> bool:
    if not match:
        return False
    match_lower = match.lower()
    # 使用 any() 和生成器表达式，一旦找到匹配项就立即返回 True
    return any(prefix in match_lower for prefix in journal_ids)


with open(unique_datasets, 'r', encoding='utf-8') as f:
    data_ids_doi = set(line.strip().lower() for line in f if line.strip().startswith("10."))
data_ids, id_details = load_all_accession_ids_efficient()
with open(reorganized_publication_dataset, 'r', encoding='utf-8') as f:
    article_data = json.load(f)
for filename in tqdm(os.listdir(pdf_directory), total=len(os.listdir(pdf_directory))):
    if filename.endswith(".pdf"):
        j += 1
        pdf_path = os.path.join(pdf_directory, filename)

        # Extract article_id from filename
        article_id = filename.split(".pdf")[0]
        # if article_id.replace("_","/") not in article_data:
        #     continue
        doc = fitz.open(pdf_path)
        text = ""
        cover = ""
        for page in doc:
            page_text = page.get_text()
            if not cover:
                cover = page_text
            elif len(cover) < 1000:
                cover += page_text
            text += page_text + "\n"

        doc.close()
        # text, ref =split_text_and_references(text)
        # print(text)

        text = normalize_text(text)
        cover = normalize_text(cover)
        text = add_period_to_heading_lines(text)
        # ref = normalize_text(ref)
        # text = re.sub(r'\s+', ' ', content)
        text = text.strip()
        # text = re.sub(r"[\u200b\u200c\u200d\uFEFF]\n|[\u200b\u200c\u200d\uFEFF]", "", text)
        text = re.sub(r'<br>', '', text)
        cover = re.sub(r'<br>', '', cover)
        text = re.sub(r'(\d+\.)\s+(\d+)', r'\1\2', text)
        text = re.sub(r'/\s', '/', text)
        # text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\\_', '_', text)
        text = fix_broken_links(text)
        # print(text)
        text = re.sub(r'(?<![\.\n])\n', ' ', text)
        # body_text = text
        cover = re.sub(r'(?<![\.\n])\n', ' ', cover)
        cover_information.append((article_id, cover))
        # full_text = text
        body_text, ref = remove_references_section_v3(text)
        data_availabilit, _ = find_data_availability_statement(text)
        if data_availabilit:
            article_data_information[article_id] = [data_availabilit]
        else:
            article_data_information[article_id] = []
        # print(data_availabilit)
        # print(body_text)

        # print(ref)
        # article_data = database_information.setdefault(article_id, {})
        # for prefix, keyword_reg in DOI_KEYWORD_RULES.items():
        #     keyword_matches = keyword_reg.finditer(body_text)
        #     for match in keyword_matches:
        #         context_list = article_data.setdefault(prefix, [])
        #         start, end = match.start(), match.end()
        #         window_start = max(0, start - 100)
        #         window_end = min(len(body_text), end + 100)
        #         context_window_text = body_text[window_start:window_end]
        #         context_list.append(context_window_text.strip())
        clustered_results = find_and_cluster_keywords(
            body_text=body_text,
            rules=DOI_KEYWORD_RULES,
            window_size=100,
            cluster_gap=50
        )

        # 如果找到了去重后的结果，就将其更新到主字典中
        if clustered_results:
            database_information.setdefault(article_id, {}).update(clustered_results)

        for pattern_name, compiled_pattern in patterns_to_find.items():
            # matches = compiled_pattern.finditer(text)
            if pattern_name == "doi" or pattern_name == "doi2":
                target_context_before = 200  # 匹配结果前取300字符
                target_context_after = 100  # 匹配结果后取200字符
                text = body_text
                matches = compiled_pattern.finditer(text)
            elif pattern_name == "doi_ref" or pattern_name == "doi_ref2":
                target_context_before = 200  # 匹配结果前取300字符
                target_context_after = 100  # 匹配结果后取200字符
                text = ref
                matches = compiled_pattern.finditer(text)
            # elif pattern_name == 'acc_id_ref':
            #     target_context_before = 200  # 匹配结果前取300字符
            #     target_context_after = 100   # 匹配结果后取200字符
            #     text = ref
            #     matches = compiled_pattern.finditer(text)
            else:
                target_context_before = 200  # 匹配结果前取300字符
                target_context_after = 100  # 匹配结果后取200字符
                text = body_text + "\n" + ref
                matches = compiled_pattern.finditer(text)
            # matches = compiled_pattern.finditer(text)
            for match in matches:
                match_text = match.group()
                match_text = re.sub(r'PDB:?\s*([1-9][A-Z0-9]{3})', r'\1', match_text)
                match_text = re.sub(r'\s+', '', match_text)
                match_text = re.sub(r'[^A-Za-z0-9]+$', '', match_text)
                if not is_balanced(match_text):
                    continue
                start, end = match.start(), match.end()

                # 获取匹配项的上下文
                context_start, context_end = find_context_boundaries(
                    text, start, end, target_context_before, target_context_after
                )
                context_chunk = text[context_start:context_end].strip()

                # 特殊处理DOI
                if pattern_name == 'doi' or pattern_name == 'doi_ref' or pattern_name == 'doi2' or pattern_name == 'doi_ref2':
                    match_text = match_text.replace(r'[-.,;:!?\/\)\]\(\[]+$', '')
                    if match_text.lower().startswith("dryad."):
                        match_text = "10.5061/" + match_text.lower()
                    elif match_text.lower().startswith("zenodo."):
                        match_text = "10.5281/" + match_text.lower()
                    elif match_text.lower().startswith("pasta/"):
                        match_text = "10.6073/" + match_text.lower()
                    #                      or journal_is_in_match(match_text)
                    elif match_text.lower().startswith("pangaea."):
                        match_text = "10.1594/" + match_text.lower()
                    if article_id.split('_')[0] in match_text or match_text.lower() in bad_ids or journal_is_in_match(
                            match_text) or len(match_text.lower().split('/')[-1]) < 4:
                        continue  # 跳过不符合要求的DOI
                    prefix = match_text.split("/")[0]
                    # print(prefix)
                    if prefix not in doi_prefix_set:
                        # print("不行",match_text)
                        continue
                    result_value = 'https://doi.org/' + match_text.lower()
                    if result_value not in data_ids_doi and match_text.lower() not in data_ids_doi:
                        continue
                else:
                    result_value = match_text
                    db_key = SPECIAL_MAPPINGS.get(pattern_name, pattern_name.split('_')[0])
                    if result_value not in data_ids:
                        continue
                    elif db_key in CONTEXT_VALIDATION_RULES:
                        window_size = CONTEXT_VALIDATION_RULES[db_key]["window_size"]
                        window_start = max(0, start - window_size)
                        window_end = min(len(text), end + window_size)

                        # 提取上下文窗口的文本
                        context_window_text = text[window_start:window_end]
                        context_regex = CONTEXT_VALIDATION_RULES[db_key]['context_regex']
                        if not context_regex.search(context_window_text):
                            # print("1")
                            continue
                if pattern_name == "doi_ref" or pattern_name == 'acc_id_ref' or pattern_name == 'doi_ref2':
                    chunks_ref.append((
                        article_id,
                        context_chunk,
                        result_value,
                        pattern_name
                    ))
                else:
                    chunks.append((
                        article_id,
                        context_chunk,
                        result_value,
                        pattern_name
                    ))
xml_files = []
for filename in tqdm(os.listdir(xml_directory), total=len(os.listdir(xml_directory))):
    if filename.endswith(".xml"):
        xml_path = os.path.join(xml_directory, filename)
        article_id = filename.split(".xml")[0]
        xml_files.append(article_id)
        text = xml2text(xml_path)
        # text = re.sub(r"[\u200b\u200c\u200d\uFEFF]\n|[\u200b\u200c\u200d\uFEFF]", "", text)
        # text = re.sub(r'<br>', '', text)
        # text = re.sub(r'(\d+\.)\s+(\d+)', r'\1\2', text)
        # text = re.sub(r'/\s+', '/', text)
        # text = re.sub(r'\n+', '\n', text)
        # text = re.sub(r'\\_', '_', text)
        # print(text)
        # body_text = text
        body_text, ref = remove_references_section_v3(text)
        full_text = text
        body_text = fix_broken_links(body_text)
        data_availabilit, _ = find_data_availability_statement(text)
        if data_availabilit:
            article_data_information[article_id].append(data_availabilit)
        ref = fix_broken_links(ref)
        # print(ref)
        target_context_before = 200  # 匹配结果前取300字符
        target_context_after = 100  # 匹配结果后取200字符
        for pattern_name, compiled_pattern in patterns_to_find.items():
            # matches = compiled_pattern.finditer(text)
            if pattern_name == "doi" or pattern_name == "doi2":
                text = body_text
                matches = compiled_pattern.finditer(text)
            elif pattern_name == "doi_ref" or pattern_name == "doi_ref2":
                text = ref
                matches = compiled_pattern.finditer(text)
            # elif pattern_name == 'acc_id_ref':
            #     text = ref
            #     matches = compiled_pattern.finditer(text)
            else:
                # continue
                text = body_text + "\n\n" + ref
                matches = compiled_pattern.finditer(text)
            # matches = compiled_pattern.finditer(text)
            for match in matches:
                match_text = match.group()
                match_text = re.sub(r'PDB:?\s*([1-9][A-Z0-9]{3})', r'\1', match_text)
                match_text = re.sub(r'\s+', '', match_text)
                match_text = re.sub(r'[^A-Za-z0-9]+$', '', match_text)
                if not is_balanced(match_text):
                    continue
                start, end = match.start(), match.end()
                if pattern_name == "doi_ref" or pattern_name == 'acc_id_ref' or pattern_name == 'doi2' or pattern_name == 'doi_ref2':
                    prev_newline = text.rfind('\n', 0, start)
                    if prev_newline == -1:  # If no newline found, start from beginning
                        prev_newline = 0
                    else:
                        prev_newline += 1  # Start after the newline character
                    prev_newline = max(prev_newline, start - target_context_before)
                    # Find the next newline
                    next_newline = text.find('\n', end)
                    if next_newline == -1:  # If no newline found, go to end of text
                        next_newline = len(text)
                    next_newline = min(next_newline, end + target_context_after)
                    # Extract the line containing the match
                    context_chunk = text[prev_newline:next_newline]
                else:
                    # 获取匹配项的上下文
                    context_start, context_end = find_context_boundaries(
                        text, start, end, target_context_before, target_context_after
                    )
                    extract_start = max(context_start, start - target_context_before)  # 往前300，但不能小于0
                    extract_end = min(context_end, end + target_context_after)  # 往后200，但不能超过全文长度
                    context_chunk = text[context_start:context_end].strip()
                # 特殊处理DOI
                if pattern_name == 'doi' or pattern_name == 'doi_ref' or pattern_name == 'doi2' or pattern_name == 'doi_ref2':
                    match_text = match_text.replace(r'[-.,;:!?\/\)\]\(\[]+$', '')
                    if match_text.lower().startswith("dryad."):
                        match_text = "10.5061/" + match_text.lower()
                    elif match_text.lower().startswith("zenodo."):
                        match_text = "10.5281/" + match_text.lower()
                    elif match_text.lower().startswith("pasta/"):
                        match_text = "10.6073/" + match_text.lower()
                    elif match_text.lower().startswith("pangaea."):
                        match_text = "10.1594/" + match_text.lower()
                    if article_id.split('_')[0] in match_text or match_text.lower() in bad_ids or journal_is_in_match(
                            match_text) or (
                            len(match_text.lower().split('/')[-1]) < 4 and not match_text.startswith('10.25326')):
                        continue  # 跳过不符合要求的DOI
                    prefix = match_text.split("/")[0]
                    # print(prefix)
                    if prefix not in doi_prefix_set:
                        continue
                    result_value = 'https://doi.org/' + match_text.lower()
                    if result_value not in data_ids_doi and match_text.lower() not in data_ids_doi:
                        continue
                else:
                    result_value = match_text
                    db_key = SPECIAL_MAPPINGS.get(pattern_name, pattern_name.split('_')[0])
                    if result_value not in data_ids:
                        continue
                    elif db_key in CONTEXT_VALIDATION_RULES:
                        window_size = CONTEXT_VALIDATION_RULES[db_key]["window_size"]
                        window_start = max(0, start - window_size)
                        window_end = min(len(text), end + window_size)

                        # 提取上下文窗口的文本
                        context_window_text = text[window_start:window_end]
                        context_regex = CONTEXT_VALIDATION_RULES[db_key]['context_regex']
                        if not context_regex.search(context_window_text):
                            continue
                if pattern_name == "doi_ref" or pattern_name == 'acc_id_ref' or pattern_name == 'doi_ref2':
                    chunks_ref.append((
                        article_id,
                        context_chunk,
                        result_value,
                        pattern_name
                    ))
                else:
                    chunks.append((
                        article_id,
                        context_chunk,
                        result_value,
                        pattern_name
                    ))
valid_chunks = set()
for article_id, context_chunk, result_value, pattern_name in chunks:
    if (article_id, result_value) not in valid_chunks:
        valid_chunks.add((article_id, result_value))
for article_id, context_chunk, result_value, pattern_name in chunks_ref:
    if (article_id, result_value) not in valid_chunks:
        valid_chunks.add((article_id, result_value))
for filename in tqdm(os.listdir(xml_directory), total=len(os.listdir(xml_directory))):
    if filename.endswith(".xml"):
        xml_path = os.path.join(xml_directory, filename)
        article_id = filename.split(".xml")[0]
        with open(xml_path, 'r', encoding='utf-8') as file:
            text = file.read()
        target_context_before = 300  # 匹配结果前取300字符
        target_context_after = 200  # 匹配结果后取200字符
        for pattern_name, compiled_pattern in patterns_to_find.items():
            matches = compiled_pattern.finditer(text)
            # matches = compiled_pattern.finditer(text)
            for match in matches:
                match_text = match.group()
                match_text = re.sub(r'PDB:?\s*([1-9][A-Z0-9]{3})', r'\1', match_text)
                match_text = re.sub(r'\s+', '', match_text)
                match_text = re.sub(r'[^A-Za-z0-9]+$', '', match_text)
                if not is_balanced(match_text):
                    continue
                start, end = match.start(), match.end()
                if pattern_name == "doi_ref" or pattern_name == 'acc_id_ref' or pattern_name == 'doi2' or pattern_name == 'doi_ref2':
                    prev_newline = text.rfind('\n', 0, start)
                    if prev_newline == -1:  # If no newline found, start from beginning
                        prev_newline = 0
                    else:
                        prev_newline += 1  # Start after the newline character
                    prev_newline = max(prev_newline, start - target_context_before)
                    # Find the next newline
                    next_newline = text.find('\n', end)
                    if next_newline == -1:  # If no newline found, go to end of text
                        next_newline = len(text)
                    next_newline = min(next_newline, end + target_context_after)
                    # Extract the line containing the match
                    context_chunk = text[prev_newline:next_newline]
                else:
                    # 获取匹配项的上下文
                    context_start, context_end = find_context_boundaries(
                        text, start, end, target_context_before, target_context_after
                    )
                    extract_start = max(context_start, start - target_context_before)  # 往前300，但不能小于0
                    extract_end = min(context_end, end + target_context_after)  # 往后200，但不能超过全文长度
                    context_chunk = text[context_start:context_end].strip()
                # 特殊处理DOI
                if pattern_name == 'doi' or pattern_name == 'doi_ref' or pattern_name == 'doi2' or pattern_name == 'doi_ref2':
                    match_text = match_text.replace(r'[-.,;:!?\/\)\]\(\[]+$', '')
                    if match_text.lower().startswith("dryad."):
                        match_text = "10.5061/" + match_text.lower()
                    elif match_text.lower().startswith("zenodo."):
                        match_text = "10.5281/" + match_text.lower()
                    elif match_text.lower().startswith("pasta/"):
                        match_text = "10.6073/" + match_text.lower()
                    elif match_text.lower().startswith("pangaea."):
                        match_text = "10.1594/" + match_text.lower()
                    if article_id.split('_')[0] in match_text or match_text.lower() in bad_ids or journal_is_in_match(
                            match_text) or (
                            len(match_text.lower().split('/')[-1]) < 4 and not match_text.startswith('10.25326')):
                        continue  # 跳过不符合要求的DOI
                    prefix = match_text.split("/")[0]
                    if prefix not in doi_prefix_set:
                        continue
                    result_value = 'https://doi.org/' + match_text.lower()
                    if result_value not in data_ids_doi and match_text.lower() not in data_ids_doi:
                        continue
                else:
                    result_value = match_text
                    db_key = SPECIAL_MAPPINGS.get(pattern_name, pattern_name.split('_')[0])
                    if result_value not in data_ids:
                        continue
                    elif db_key in CONTEXT_VALIDATION_RULES:
                        window_size = CONTEXT_VALIDATION_RULES[db_key]["window_size"] * 2
                        window_start = max(0, start - window_size)
                        window_end = min(len(text), end + window_size)

                        # 提取上下文窗口的文本
                        context_window_text = text[window_start:window_end]
                        context_regex = CONTEXT_VALIDATION_RULES[db_key]['context_regex']
                        if not context_regex.search(context_window_text):
                            # print("1")
                            continue
                if (article_id, result_value) in valid_chunks:
                    continue
                valid_chunks.add((article_id, result_value))
                chunks.append((
                    article_id,
                    context_chunk,
                    result_value,
                    pattern_name
                ))
# chunks = chunks + chunks_ref
df_chunks = pd.DataFrame(chunks, columns=['article_id', 'context_chunk', 'result_value', 'pattern_name'])

# 保存为CSV文件
df_chunks.to_csv('/kaggle/working/chunks.csv', index=False, encoding='utf-8')
chunks_processed = []
for article_id, context_chunk, result_value, pattern_name in chunks:
    if result_value.startswith('https://doi.org/'):
        chunks_processed.append((article_id, context_chunk, result_value, pattern_name))
    elif article_id in xml_files:
        chunks_processed.append((article_id, context_chunk, result_value, pattern_name))
chunks = chunks_processed
print(f"\n在所有文件中总共找到了 {len(chunks)} 个匹配项。")

##Ask LLM to classify DOI links
##Use logits-processor-zoo MultipleChoiceLogitsProcessor to enforce LLM choose between classes.
print('10.5281' not in doi_prefix_set)


def filter_chunks_by_majority_id(chunks, id_details_dict):
    """
    根据每组 article_id 中多数派的 PMCID 或 EXTID 过滤 chunks。
    修改：保留 result_value 以 https 开头的项，且不对其进行过滤。

    Args:
        chunks: 包含 (article_id, context_chunk, result_value, pattern_name) 元组的列表。
        id_details_dict: 由 load_all_accession_ids_efficient 返回的 id_details 字典。
                        结构: {id: {'pmc_ids': set(), 'ext_ids': set()}}

    Returns:
        list: 过滤后的 chunks 列表。
    """
    if not chunks:
        return []

    # 1. 按 article_id 分组 (但这次不预先过滤 HTTPS)
    grouped_chunks = defaultdict(list)
    https_chunks = []  # 新增：用于存储 HTTPS 开头的 chunks

    for item in chunks:
        article_id, context_chunk, result_value, pattern_name = item
        # 检查 result_value 是否以 https 开头
        if result_value and isinstance(result_value, str) and result_value.lower().startswith("https"):
            # 如果是，添加到专门的列表中
            https_chunks.append(item)
            # print(f"暂存 HTTPS 项: {item}")
        else:
            # 如果不是 HTTPS 开头，则按 article_id 分组 (用于后续过滤)
            grouped_chunks[article_id].append(item)

    filtered_chunks = []
    # 首先，将所有 HTTPS 开头的 chunks 添加到结果中
    filtered_chunks.extend(https_chunks)
    print(f"保留了 {len(https_chunks)} 个 result_value 以 'https' 开头的项。")

    processed_groups_count = 0

    # 2. 遍历每个需要处理的分组 (非 HTTPS 的项)
    for article_id, group_items in grouped_chunks.items():
        if not group_items:
            continue

        # 3. 收集该组所有 result_value 对应的 PMCID 和 EXTID
        all_pmcids = []
        all_extids = []
        item_details_map = {}  # 临时存储每个 item 及其对应的 details

        for item in group_items:
            _, _, result_value, _ = item
            # 使用 result_value 查找详细信息
            details = id_details_dict.get(result_value, {'pmc_ids': set(), 'ext_ids': set()})
            item_details_map[item] = details
            all_pmcids.extend(list(details['pmc_ids']))
            all_extids.extend(list(details['ext_ids']))

        # 4. 统计并找出出现频率最高的 PMCID 和 EXTID
        pmcid_counter = Counter(all_pmcids)
        extid_counter = Counter(all_extids)

        most_common_pmcid = pmcid_counter.most_common(1)[0][0] if pmcid_counter else None
        most_common_extid = extid_counter.most_common(1)[0][0] if extid_counter else None

        # 如果没有找到任何 PMCID 或 EXTID，保留所有该组的非 HTTPS 项
        if most_common_pmcid is None and most_common_extid is None:
            print(f"警告: 分组 {article_id} 中没有找到任何关联的 PMCID 或 EXTID，保留所有非-HTTPS 项。")
            filtered_chunks.extend(group_items)
            processed_groups_count += 1
            continue

        # print(f"分组 {article_id} - 最频繁 PMCID: {most_common_pmcid}")
        # print(f"分组 {article_id} - 最频繁 EXTID: {most_common_extid}")

        # 5. 再次遍历该组，根据多数派 ID 进行过滤 (仅针对非 HTTPS 项)
        group_kept_items = []
        for item in group_items:
            details = item_details_map.get(item, {'pmc_ids': set(), 'ext_ids': set()})
            item_pmcids = details['pmc_ids']
            item_extids = details['ext_ids']

            # 判断保留条件
            keep_item = False
            if most_common_pmcid and most_common_pmcid in item_pmcids:
                keep_item = True
            elif most_common_extid and most_common_extid in item_extids:
                keep_item = True

            if keep_item:
                group_kept_items.append(item)

        # print(f"分组 {article_id} 处理完成，保留 {len(group_kept_items)} / {len(group_items)} 个非-HTTPS 项。")
        filtered_chunks.extend(group_kept_items)
        processed_groups_count += 1

    print(f"总共处理了 {processed_groups_count} 个需要过滤的 article_id 分组。")
    print(f"过滤后剩余 chunks 总数: {len(filtered_chunks)} (原始总数: {len(chunks)})")
    return filtered_chunks


chunks = filter_chunks_by_majority_id(chunks, id_details)

df_chunks = pd.DataFrame(chunks, columns=['article_id', 'context_chunk', 'result_value', 'pattern_name'])

# 保存为CSV文件
df_chunks.to_csv('/kaggle/working/filter_chunks.csv', index=False, encoding='utf-8')

if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    label_df = pd.read_csv(labels_dir)
    # 过滤掉 'Missing' 类型的标签
    label_df = label_df[label_df['type'] != 'Missing'].reset_index(drop=True)

    # 1. 提取标签数据中的 (article_id, dataset_id) pairs
    # 确保 ID 是字符串类型，处理可能的 NaN 值
    label_df['article_id_str'] = label_df['article_id'].fillna('NaN').astype(str)
    label_df['dataset_id_str'] = label_df['dataset_id'].fillna('NaN').astype(str)

    # 创建一个包含所有正确 (article_id, dataset_id) pairs 的集合
    # 使用元组 (article_id, dataset_id) 作为集合元素
    true_pairs = set(zip(label_df['article_id_str'], label_df['dataset_id_str']))
    # 或者如果你想确保 dataset_id 不是 'Missing' 且两个 ID 都存在，可以在上面的过滤后进行：
    # true_pairs = set(zip(label_df['article_id'].astype(str), label_df['dataset_id'].astype(str)))

    # 2. 提取检测结果中的 (article_id, dataset_id) pairs
    # 假设 chunks 和 chunks_ref 中的元素结构是 (start, end, dataset_id, article_id)
    # 你需要根据实际结构调整索引 item[2] 是 dataset_id, item[3] 是 article_id
    # 同样确保它们是字符串
    try:
        detected_pairs = {(str(item[0]), str(item[2])) for item in chunks + chunks_ref}  # (article_id, dataset_id)
    except IndexError:
        print("警告：chunks 或 chunks_ref 中的元素结构与预期不符，无法提取 (article_id, dataset_id) pair。")
        detected_pairs = set()
    # 如果结构是 (dataset_id, article_id, ...), 则使用:
    # detected_pairs = {(str(item[1]), str(item[0])) for item in chunks + chunks_ref}

    print(f"\n在所有文件中总共找到了 {len(detected_pairs)} 个 (article_id, dataset_id) 匹配项。")

    # 3. 计算差异
    # 多出的检测结果：在 detected 中但不在 true 中
    extra_detections = detected_pairs - true_pairs
    # 缺少的检测结果：在 true 中但不在 detected 中
    missing_detections = true_pairs - detected_pairs

    # 4. 输出结果
    print(f"多出的检测结果数量 (False Positives): {len(extra_detections)}")
    if extra_detections:
        # 打印前100个多出的 pairs 作为示例
        # extra_detections 是一个元组集合，可以直接转换为字符串列表
        sorted_extra = sorted(list(extra_detections))
        print("多出的内容示例 (article_id, dataset_id):", "\n".join([str(pair) for pair in list(sorted_extra)[:400]]))

    print(f"\n缺少的检测结果数量 (False Negatives): {len(missing_detections)}")
    if missing_detections:
        # 打印前100个缺少的 pairs 作为示例
        sorted_missing = sorted(list(missing_detections))
        print("缺少的内容示例 (article_id, dataset_id):", "\n".join([str(pair) for pair in list(sorted_missing)[:100]]))

# MODEL_DIRECTORY = '/kaggle/input/scibert-finetuning/scibert_classifier_model_balanced_0710'
# classifier = BatchCitationClassifier(model_path=MODEL_DIRECTORY)
# prediction_results = classifier.predict_batch(chunks)
# print("\n" + "="*20 + " 批量预测结果 " + "="*20)
# answers = [None] * len(chunks)
# for i, result in enumerate(prediction_results):
#     answers[i] = result['predicted_label'] if result['predicted_label'] != "Unknown" else None
# print("\n" + "="*55)
# del classifier
# torch.cuda.empty_cache()
# import gc
# gc.collect()
answers = ["Secondary"] * len(chunks)
# chunks_ref_pro = []
# body_length = len(chunks)
# for i in range(body_length):
#     if chunks[i][2].startswith("https://doi.org/"):
#         answers[i] = "Secondary"
# chunks.extend(chunks_ref)
# answers.extend(["Secondary"] * len(chunks_ref))
body_length = len(chunks)
for i in range(body_length):
    if chunks[i][2].startswith("https://doi.org/"):
        answers[i] = "Secondary"
    elif chunks[i][2].startswith("SAMN"):
        answers[i] = "Primary"
    else:
        answers[i] = "Secondary"
chunks.extend(chunks_ref)
answers.extend(["Secondary"] * len(chunks_ref))
# for article_id, academic_text, dataset_id,pattern_name in chunks_ref:
#     chunks_ref_pro.append((article_id, academic_text, dataset_id,pattern_name))
#     answers.append("Secondary")
# chunks = chunks + chunks_ref_pro

df_answers = pd.DataFrame(answers, columns=['answers'])

# 保存为CSV文件
df_answers.to_csv('/kaggle/working/answers.csv', index=False, encoding='utf-8')


if LOCAL:
    model_path = "/root/autodl-tmp/Qwen2.5-32B-Insturct-AWQ"
    # model_path = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
    # model_path = "/kaggle/input/qwen3-30b-a3b-instruct-2507-awq/transformers/default/1"
else:
    model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = vllm.LLM(
    model_path,
    # quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=4096,
    disable_log_stats=True,
    enable_prefix_caching=True
)
tokenizer = llm.get_tokenizer()

SYS_PROMPT_AUTHOR_INFORMATION = """
You are a highly efficient and precise author extraction tool. Your sole task is to identify and extract all main authors of a paper from the provided cover page text.

=== Core Principles ===
- **Find the Block, Then Extract:** Your main goal is to first identify the entire *block* of text containing the author list. Once the block is identified, you MUST extract *all* names from it.
- **Signals Validate the Block, Not the Individual:** Superscripts (¹, *) are powerful clues that you have found the correct *group* of names. An individual name does not need a signal to be an author if it is part of that group.

=== Step-by-Step Search Strategy ===
Follow this process to ensure accuracy:

1.  **Locate Anchors:** First, identify the main article **title** and the **abstract**. The author list is almost always located in the space between these two sections.
2.  **Identify the Candidate Author Block:** Within that region, search for a block of text that consists of personal names. This block is your primary candidate.
3.  **Confirm the Block Using Signals:** Verify that this block is the author list by checking if **at least some of the names** are associated with strong signals:
    - **Superscripts:** Numbers (¹, ², ³) or symbols (*, †, ‡) immediately following a name.
    - **Affiliation Markers:** Names followed by letters (a, b, c).
    - **Corresponding Author Indicators:** An asterisk (*) or the explicit phrase `Corresponding author:`.
4.  **Extract ALL Names from the Confirmed Block:** **This is a critical step.** Once you are confident you have the author block, extract every name within it. Do not omit names just because they lack a superscript or other marker.
5.  **Apply Exclusion Rules (Crucial):** To avoid errors, you MUST IGNORE names found in the following contexts:
    - **In-text Citations:** Any names inside parentheses, e.g., `(Smith et al., 2022)`.
    - **References/Bibliography:** Any names in a reference list.
    - **Labeled Non-Authors:** Names explicitly labeled as 'Editor', 'Reviewer', or found in sections like 'Acknowledgements'.

=== Formatting Rules ===
1.  Extract only the full names of the authors.
2.  **Remove all extra characters,** including superscript numbers (¹, ², a, *), affiliation markers, degrees, and the word "and" before the last author.
3.  List all author names, separated by commas.
4.  Place the final comma-separated list of names inside an `<authors>` tag.

=== Examples ===
Input Text:
"A Novel Approach to Machine Learning
John A. Smith¹, Jane B. Doe²*, and Michael C. Lee¹
¹Department of Computer Science, University of Innovation
²Institute for Advanced Studies"
Output:
<authors>John A. Smith, Jane B. Doe, Michael C. Lee</authors>

Input Text:
"Cellular Mechanisms of Memory Formation
ANNA KOWALSKI¹, PIOTR NOWAK, and JANE DOE¹'²*
¹Institute of Neuroscience, ²Center for Advanced Brain Studies
*Corresponding author: j.doe@email.com
Abstract: Memory is a complex process..."
Output:
<authors>ANNA KOWALSKI, PIOTR NOWAK, JANE DOE</authors>

Input Text:
"Nature Communications | (2025) 16:1234 | https://doi.org/10.1038/s41467-025-12345-x
ARTICLE
Deep learning for climate model analysis
Carlos de la Cruz¹'², Wei Zhang¹*, and Jane Smith³
¹Climate Research Institute, ²Department of Physics, ³Data Science Center
Abstract: In this paper, we... The work of (Jones, 2021) is relevant..."
Output:
<authors>Carlos de la Cruz, Wei Zhang, Jane Smith</authors>

=== Instruction ===
Analyze the following cover information using the search strategy above. Find the main authors of the article and output ONLY the `<authors>` tag containing their names.
""".strip()
prompts = []
for i, item in enumerate(cover_information):
    article_id, cover_text = item
    messages = [
        {"role": "system", "content": SYS_PROMPT_AUTHOR_INFORMATION},
        {"role": "user", "content": f"Cover Information:{cover_text[:2000]}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    prompts.append(prompt)
print(len(prompts[0]))
outputs = llm.generate(
    prompts,
    vllm.SamplingParams(
        seed=42,
        skip_special_tokens=True,
        max_tokens=96,
        temperature=0.1
    ),
    use_tqdm=True
)
responses = [output.outputs[0].text for output in outputs]

author_information = {}

for i, response in enumerate(responses):
    author_information[cover_information[i][0]] = response.split("</authors>")[0].replace("<authors>","")

