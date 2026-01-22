from CReDEsPydanticModel import MultiQARequestFactory
from typing import Dict,List,Any,Type,Optional
from pydantic import BaseModel,Field,create_model,ValidationError,ConfigDict
import json
import time
from functools import wraps
import os 
from instructor.core import InstructorRetryException
import re
import json_repair

def get_processed_CDEs_base_CDE(CDEs:Dict[str,Dict[str,List[Dict[str,Any]]]],projects_name:str, need_grouped_cde_dict:Dict[str,List[str]]):
    """获取需要分组的CDEs"""
    need_grouped_cde_item = {json_key:[] for json_key in need_grouped_cde_dict.keys()}
    non_grouped_cde_item = []

    target_project_cdes = CDEs.get(projects_name, [])
    target_project_cdes_name = [cdes.get("CReDEs_metadata_name") or cdes.get("label_name") for cdes in target_project_cdes]


    for key,value in need_grouped_cde_dict.items():
        try:
            for cde_name in value:
                if cde_name in target_project_cdes_name:
                    # 找到对应的 CDE 对象（dict）
                    matched_item = next(
                                    (c for c in target_project_cdes
                                    if (c.get("CReDEs_metadata_name") or c.get("label_name")) == cde_name),
                                    None
                                )
                    if matched_item:
                        need_grouped_cde_item[key].append(matched_item)
        except Exception as e:
            print(f"Error occurred while processing {key}: {e}")
            continue

    # 非分组项
    grouped_names = set(sum(need_grouped_cde_dict.values(), []))
    for cde in target_project_cdes:
        # CReDEs_metadata_name 可能为空
        cde_name = ""
        if "CReDEs_metadata_name" in cde:
            cde_name = cde["CReDEs_metadata_name"]
        else:
            cde_name = cde.get("label_name")
            
        if cde_name not in grouped_names:
            non_grouped_cde_item.append(cde)

    return need_grouped_cde_item, non_grouped_cde_item
    

def create_multi_qa_model_for_grouped_cde(grouped_cde_item:Dict[str,List[Dict[str,Any]]]) -> Dict[str,Type[BaseModel]]:
    """为分组标签创建多QA请求模型"""
    grouped_models = {}
    for group_name, cde_list in grouped_cde_item.items():
        multi_qa_model = MultiQARequestFactory.create_multi_qa_model(cde_list)
        grouped_model = create_model(
                    group_name,
                    **{
                        group_name: (
                            List[multi_qa_model],
                            Field(
                                ...,
                                description=f"这是 {group_name} 分组的多QA数据列表，文中可能包含多个符合条件的组别，请全部填写。"
                            ),
                        )
                    }
                )
        grouped_models[group_name] = grouped_model

    return grouped_models

def create_multi_qa_model_for_non_grouped_cde(non_grouped_labels:List[Dict[str,Any]]) -> Type[BaseModel]:
    """为非分组标签创建多QA请求模型"""
    multi_qa_model = MultiQARequestFactory.create_multi_qa_model(non_grouped_labels)
    return multi_qa_model

def load_json(file_path:str) -> Dict[str,Any]:
    with open(file_path,"r") as f:
        data = json.load(f)
    return data

def save_data(data:Dict[str,Any],dir_name:str,file_name:str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_path = os.path.join(dir_name,file_name)
    with open(file_path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


def get_prompt(basic_prompt:str,source_text:str,annotation_text:Optional[str]=None) -> str:
    prompt = basic_prompt.format(source_text=source_text,annotation_text=annotation_text)
    return prompt


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行耗时: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

def append_to_jsonl(file_path: str, data: dict):
    """将单条推理结果以 JSONL 格式追加写入文件"""
    with open(file_path, 'a', encoding='utf-8') as f:
        line = json.dumps(data, ensure_ascii=False)
        f.write(line + '\n')


def convert_jsonl_to_json(jsonl_path: str, json_path: str):
    """
    读取 JSONL 记录文件并转换成标准的 JSON 列表格式。
    用于在项目结束时生成最终可交付的结果文件。
    """
    data_list = []
    if not os.path.exists(jsonl_path):
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    
    print(f"[✓] 转换完成，最终结果已保存至: {json_path}")


def get_processed_article_ids(file_path: str) -> set:
    """读取已处理成功的 article_id，用于实现断点续传"""
    processed_ids = set()
    if not os.path.exists(file_path):
        return processed_ids

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "article_id" in data:
                    processed_ids.add(data["article_id"])
            except:
                continue
    return processed_ids

import re
import json
import json_repair  # 需安装: pip install json-repair

def process_error_last_completion(raw_content: str) -> any:
    """
    处理错误情况下的最后生成内容
    """
    if not raw_content:
        return {}

    if "</think>" in raw_content:
        clean_content = raw_content.split("</think>")[-1]
    else:
        clean_content = raw_content
    
    clean_content = clean_content.strip()

    candidate_str = ""
    
    # 策略 A: 在 clean_content 中查找代码块 (注意：必须查 clean_content)
    # 优化正则：忽略大小写，且 json 标识符变为可选
    code_block_pattern = r"```(?:json)?\s*(.*?)```"
    code_blocks = re.findall(code_block_pattern, clean_content, re.DOTALL | re.IGNORECASE)

    if code_blocks:
        candidate_str = code_blocks[-1].strip()
    else:
        # 策略 B: 没有代码块，尝试提取首尾花括号的内容
        start_idx = clean_content.find("{")
        end_idx = clean_content.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate_str = clean_content[start_idx : end_idx + 1]
        else:
            candidate_str = clean_content

    try:
        return json.loads(candidate_str)
    except Exception:
        try:
            repaired_str = json_repair.repair_json(candidate_str)
            return json.loads(repaired_str)
        except Exception as e:
            return {}
        
def get_pydantic_model(cdes: Dict, project_name: str, data_config: Dict):
    """
    动态构建 Pydantic 模型，用于约束 LLM 的结构化输出
    显性推理过程:reasoning section
    答案提取部分:answers section
    """
    project_config = data_config[project_name]
    need_grouped_cde_dict = project_config["need_grouped_cde_dict"]

    need_grouped_item, non_grouped_item = get_processed_CDEs_base_CDE(
        cdes, project_name, need_grouped_cde_dict
    )

    # 如果存在分组字段
    if len(need_grouped_item) > 0:
        final_fields = {}
        # answer_final_fields = {}
        grouped_models = create_multi_qa_model_for_grouped_cde(need_grouped_item)
        non_grouped_model = create_multi_qa_model_for_non_grouped_cde(non_grouped_item)

        # 提取分组字段
        for g_name, model in grouped_models.items():
            field_info = model.model_fields[g_name]
            final_fields[g_name] = (
                field_info.annotation, 
                Field(..., description=field_info.description)
            )

        # 提取非分组字段
        for f_name, field in non_grouped_model.model_fields.items():
            annotation = field.annotation
            info = field.field_info if hasattr(field, "field_info") else field
            final_fields[f_name] = (annotation, info)

        return create_model(
            "FinalModel", 
            __config__=ConfigDict(populate_by_name=True), 
            **final_fields
        )
    else:
        non_grouped_model = create_multi_qa_model_for_non_grouped_cde(non_grouped_item)
        return non_grouped_model

  

