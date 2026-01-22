import os
import json
import asyncio
from typing import Dict, List, Any, Type, Tuple
import instructor
from pydantic import BaseModel, create_model, Field, ConfigDict,ValidationError
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from instructor.core import InstructorRetryException
from instructor import Mode
from CReDEsPydanticModel import CReDEs_FileProcessor,ReasoningStep,GroupedAnalysis,NonGroupedAnalysis,SynthesisAnalysis
from params import BAISC_PROMPT, NEW_SYSTEM_PROMPT
from utils import (
    get_processed_CDEs_base_CDE,
    create_multi_qa_model_for_grouped_cde,
    create_multi_qa_model_for_non_grouped_cde,
    load_json,
    get_prompt,
    append_to_jsonl,
    convert_jsonl_to_json,
    get_processed_article_ids,
    process_error_last_completion
)

# ---------------------------------------------------------
# 1. 基础工具函数 (文件 I/O 与 Client 初始化)
# ---------------------------------------------------------

def init_async_client(base_url: str, api_key: str):
    """初始化被 instructor 封装的异步 OpenAI 客户端"""
    client = instructor.from_openai(
        AsyncOpenAI(base_url=base_url, api_key=api_key),
        mode=Mode.MD_JSON
    )
    return client

# ---------------------------------------------------------
# 2. 模型构建与数据预处理
# ---------------------------------------------------------

# get_pydantic_model_Reasoning_Model(
def get_pydantic_model_Reasoning_Model(cdes: Dict, project_name: str, data_config: Dict):
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

    answer_final_fields = {}
    if len(need_grouped_item) > 0:
        grouped_models = create_multi_qa_model_for_grouped_cde(need_grouped_item)
        non_grouped_model = create_multi_qa_model_for_non_grouped_cde(non_grouped_item)
        
        # 组装答案字段
        for g_name, model in grouped_models.items():
            field_info = model.model_fields[g_name]
            answer_final_fields[g_name] = (field_info.annotation, Field(..., description=field_info.description))
        for f_name, field in non_grouped_model.model_fields.items():
             # 注意：处理 field_info 的兼容性
            info = field.field_info if hasattr(field, "field_info") else field
            answer_final_fields[f_name] = (field.annotation, info)
    else:
        # 只有非成组
        non_grouped_model = create_multi_qa_model_for_non_grouped_cde(non_grouped_item)
        for f_name, field in non_grouped_model.model_fields.items():
            info = field.field_info if hasattr(field, "field_info") else field
            answer_final_fields[f_name] = (field.annotation, info)

    AnswersModel = create_model(
        "AnswersSectionModel",
        __config__=ConfigDict(populate_by_name=True),
        **answer_final_fields
    )
    # 组装Reasoning section --> 按照非成组-> 成组 -> 结合分析 -> 填写最终答案
    final_fields = {}
    # Step 1: 分析非成组
    final_fields["phase1_non_grouped_analysis"] = (
        NonGroupedAnalysis,
        Field(..., description="第一阶段：分析所有独立、全局的字段。")
    )
    # Step 2: 分析成组 (如果有)
    if len(need_grouped_item) > 0:
        final_fields["phase2_grouped_analysis"] = (
            GroupedAnalysis,
            Field(..., description="第二阶段：结合原文及标注数据分析所有成组的CDE。")
        )
        # Step 3: 整合校验
        final_fields["phase3_synthesis"] = (
            SynthesisAnalysis,
            Field(..., description="第三阶段：综合前两步分析，解决冲突，进行最终数据完整性校验。")
        )

        # Step 4: 最终填空
        final_fields["final_answers"] = (
            AnswersModel,
            Field(..., description="最终阶段：基于上述三个阶段的推理结果，填充符合Schema的JSON数据。")
        )
    else:
        final_fields["phase2_synthesis"] = (
            SynthesisAnalysis,
            Field(..., description="第二阶段：综合第一步分析，解决冲突，进行最终数据完整性校验。")
        )
        final_fields["final_answers"] = (
            AnswersModel,
            Field(..., description="最终阶段：基于上述两个阶段的推理结果，填充符合Schema的JSON数据。")
        )

    return create_model(
        "ThreeStageExtractionModel", 
        __config__=ConfigDict(populate_by_name=True), 
        **final_fields
    )
# get_pydantic_model_nonreasoning_section
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


def prepare_inference_data(project_name: str, data_config: Dict, processed_ids: set):
    """加载原始数据并过滤掉已处理的部分 (断点续传)"""
    project_config = data_config[project_name]
    raw_data = load_json(project_config["entity_inferecne_path"])
    
    prompt_list = []
    skipped_count = 0

    for item in raw_data:
        article_id = item.get("article_id", "")
        
        # 核心断点逻辑
        if article_id in processed_ids:
            skipped_count += 1
            continue

        prompt = get_prompt(
            BAISC_PROMPT, 
            # v6
            # BAISC_PROMPT_temp,
            item.get("text", ""), 
            item.get("pred_entities_text", "")
        )
        
        prompt_list.append({
            "article_id": article_id, 
            "text": item.get("text", ""),
            "prompt": prompt
        })

    if skipped_count > 0:
        print(f"[*] 断点续传: 自动跳过已处理的 {skipped_count} 条数据")
        
    return prompt_list


# ---------------------------------------------------------
# 3. 核心异步推理逻辑
# ---------------------------------------------------------

async def call_llm_with_retry(
    client, 
    model: str, 
    prompt: str, 
    model_class: Type[BaseModel], 
    temperature: float, 
    max_retries: int = 2,
    timeout: int = 300
):
    try:
        return await asyncio.wait_for(
            client.create(
                model=model, 
                response_model=model_class, 
                temperature=temperature, 
                max_retries=max_retries,  # 启用 instructor 内部重试
                max_tokens=16384,
                messages=[
                    {"role": "system", "content": NEW_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True}
                },
            ),
            timeout=timeout
        )
    except Exception as e:
        # 1. 初始化默认值，防止 UnboundLocalError
        raw_content = None
        is_raw_failure = False

        # 2. 处理 Instructor 重试耗尽异常 (校验失败)
        if isinstance(e, InstructorRetryException):
            is_raw_failure = True # 确定是生成了内容但校验不过
            
            # 安全地提取 last_completion
            if hasattr(e, "last_completion") and e.last_completion and e.last_completion.choices:
                message = e.last_completion.choices[0].message
                if message.tool_calls:
                    raw_content = message.tool_calls[0].function.arguments
                elif message.content:
                    raw_content = message.content

        # 3. 处理超时
        elif isinstance(e, asyncio.TimeoutError):
            is_raw_failure = False

        # 4. 统一返回错误结构
        return {
            "raw_response": raw_content, 
            "error_msg": str(e), 
            "is_raw_failure": is_raw_failure,
            "error_type": type(e).__name__
        }
        



async def run_batch_inference(
    model_config: Dict, 
    pydantic_model: Type[BaseModel], 
    prompt_list: List[Dict], 
    client, 
    jsonl_path: str, 
    max_concurrency: int = 16
):
    """并发调度核心函数，支持实时写入"""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def single_task(p):
        async with semaphore:
            article_id = p["article_id"]
            
            # 执行 LLM 推理
            resp = await call_llm_with_retry(
                client=client, 
                model=model_config["model_path"], 
                prompt=p["prompt"], 
                model_class=pydantic_model, 
                temperature=model_config["temperature"]
            )
            
            # 格式化结果
            result_item = {
                "article_id": article_id, 
                "text": p["text"], 
                "answers": {},
                "error_msg": {},
                "error_type": ""
            }

            if isinstance(resp, BaseModel):
                # result_item["answers"] = resp.model_dump()
                result_item["answers"] = resp.model_dump(mode='json')
            else:
                last_completion = resp.get("raw_response")
                if resp.get("is_raw_failure") and last_completion:
                    # 如果是校验失败但有内容，尝试存储最后的生成内容
                    result_item["answers"] = process_error_last_completion(last_completion)
                result_item["error_msg"] = {
                    "_status": "failed", 
                    "raw": resp.get("raw_response"), 
                    "msg": resp.get("error_msg")
                }
                result_item["error_type"] = resp.get("error_type","")
            
            # 每完成一条，立即追加写入 JSONL 子文件夹
            append_to_jsonl(jsonl_path, result_item)
            return result_item

    tasks = [asyncio.create_task(single_task(p)) for p in prompt_list]
    
    results = []
    for coro in tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc="推理进度"):
        results.append(await coro)
    
    return results


# ---------------------------------------------------------
# 4. 项目流水线调度
# ---------------------------------------------------------

async def run_single_project_pipeline(project_name: str, local_config: Dict, client):
    """管理单个项目的全生命周期：路径准备 -> 断点扫描 -> 推理 -> 格式转换"""
    print(f"\n" + "="*50)
    print(f">>> 启动项目: {project_name}")
    
    # 路径管理逻辑
    base_store_dir = local_config["store_transfer_data_path_dir"]
    record_dir = os.path.join(base_store_dir, "jsonl_records")
    
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    
    # 临时文件存放在子目录，最终文件存放在主目录
    jsonl_path = os.path.join(record_dir, f"{project_name}_temp.jsonl")
    final_json_path = os.path.join(base_store_dir, f"{project_name}_transfer_data.json")

    # 1. 扫描已存在的断点数据
    processed_ids = get_processed_article_ids(jsonl_path)
    
    # 2. 准备 Pydantic 模型和过滤后的推理数据
    data_info_config = local_config["data"]["transfer_data_info"]
    file_proc = CReDEs_FileProcessor(
        local_config["data"]['label_mapping_path'], 
        local_config["data"]['CReDEs_mapping_path']
    )
    mapping_dict = file_proc.get_CReDEs_mapping_dict()

    target_model = get_pydantic_model(mapping_dict, project_name, data_info_config)
    inference_tasks = prepare_inference_data(project_name, data_info_config, processed_ids)

    # 3. 执行核心推理流程
    if not inference_tasks:
        print(f"--- 项目 {project_name} 已全部处理完成，跳过推理。")
    else:
        await run_batch_inference( 
            model_config=local_config["model_config"], 
            pydantic_model=target_model, 
            prompt_list=inference_tasks, 
            client=client, 
            jsonl_path=jsonl_path, 
            max_concurrency=20
        )

    # 4. 无论是否新跑了数据，均将 JSONL 转换为标准 JSON
    convert_jsonl_to_json(jsonl_path, final_json_path)


# ---------------------------------------------------------
# 5. 主程序入口
# ---------------------------------------------------------

async def main_async():
    """主异步控制器"""
    # 加载本地配置文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
        
    local_inference_config = load_json(config_path)["local_inference"]
    
    # 初始化全局 Client
    client = init_async_client(
        base_url=local_inference_config["model_config"]["base_url"], 
        api_key=local_inference_config["model_config"]["api_key"]
    )
    
    # 定义需要运行的项目列表
    # dg
    projects = ["病理诊断", "检查", "介入治疗", "手术记录", "日常病程", "入院记录"]

    # jy
    # projects = [ "病理检查", "个人史", "婚育史", "既往史", "手术记录", "现病史", "影像检查", "专科检查"]

    # xk
    # projects = [ "既往史","家族史", "诊断", "病理", "物理检查", "个人史", "婚育史", "过敏史", "现病史"]

    
    try:
        for project in projects:
            await run_single_project_pipeline(project, local_inference_config, client)
    finally:
        # 显式关闭连接池，防止出现 Event loop closed 错误
        await client.client.close()
        print("\n" + "="*50)
        print("[!] 所有推理任务执行结束，客户端已安全关闭。")


def main():
    """主入口函数"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[!] 手动中断运行。")


if __name__ == "__main__":
    main()