import instructor
from pydantic import BaseModel, create_model, Field, ConfigDict
from openai import AsyncOpenAI,OpenAI
from CReDEsPydanticModel import CReDEs_FileProcessor
from typing import Dict, List, Any, Type
import os
import asyncio
from instructor.core import InstructorRetryException
from tqdm.asyncio import tqdm_asyncio
from utils import (
    get_processed_CDEs_base_CDE,
    create_multi_qa_model_for_grouped_cde,
    create_multi_qa_model_for_non_grouped_cde,
    load_json,
    save_data,
    get_prompt,
    timeit
)
from params import BAISC_PROMPT_temp


# -----------------------------
# 1. 初始化异步 Client（正确）
# -----------------------------
def init_async_client(base_url: str, api_key: str):
    return instructor.from_openai(
        AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
    )

# -----------------------------
# 2. 构建 pydantic 模型（保持你的原逻辑）
# -----------------------------
def get_pydantic_model(
    cdes: Dict[str, Dict[str, List[Dict[str, Any]]]],
    project_name: str,
    data_config: Dict[str, Any]
):
    project_config = data_config[project_name]
    need_grouped_cde_dict = project_config["need_grouped_cde_dict"]

    need_grouped, non_grouped = get_processed_CDEs_base_CDE(
        cdes, project_name, need_grouped_cde_dict
    )

    if need_grouped:
        final_fields = {}

        grouped_models = create_multi_qa_model_for_grouped_cde(need_grouped)
        non_grouped_model = create_multi_qa_model_for_non_grouped_cde(non_grouped)

        for grouped_name, model in grouped_models.items():
            field = model.model_fields[grouped_name]
            final_fields[grouped_name] = (
                field.annotation,
                Field(..., description=field.description)
            )

        for field_name, field in non_grouped_model.model_fields.items():
            if hasattr(field, "field_info"):
                final_fields[field_name] = (field.annotation, field.field_info)
            else:
                final_fields[field_name] = (field.annotation, field)

        return create_model(
            "FinalQAResponseModel",
            __config__=ConfigDict(populate_by_name=True),
            **final_fields
        )

    return create_multi_qa_model_for_non_grouped_cde(non_grouped)


# -----------------------------
# 3. 数据准备
# -----------------------------
def process_data(project_name: str, data_config: Dict[str, Any]):
    project_config = data_config[project_name]
    inference_data = load_json(project_config["entity_inferecne_path"])

    prompts = []
    for item in inference_data:
        prompt = get_prompt(
            BAISC_PROMPT_temp,
            item.get("input", ""),
            item.get("output", "")
        )
        prompts.append({
            "article_id": item.get("article_id", ""),
            "text": item.get("input", ""),
            "prompt": prompt
        })
    return prompts


# -----------------------------
# 4. 单次 LLM 调用（已修复 NoneType 报错）
# -----------------------------
async def call_llm_with_retry(
    client,
    model: str,
    prompt: str,
    model_class: Type[BaseModel],
    temperature: float,
    max_retries: int = 2,
    timeout: int = 600,
):
    # 初始化最后一次的异常信息，防止循环结束时无具体错误
    last_error = "Unknown error"

    for retry in range(max_retries):
        try:
            model_obj, completion = await asyncio.wait_for(
                client.chat.completions.create_with_completion(
                    model=model,
                    response_model=model_class,
                    temperature=temperature,
                    max_retries=2,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你必须严格输出 JSON，且必须严格匹配 schema。"
                                "禁止输出解释、禁止输出自然语言。"
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
                ),
                timeout=timeout
            )

            return {
                "ok": True,
                "model": model_obj,
                "completion": completion,
                "error": None
            }

        except asyncio.TimeoutError:
            last_error = f"Timeout after {max_retries} retries"
            if retry == max_retries - 1:
                return {
                    "ok": False,
                    "model": None,
                    "completion": None,
                    "error": last_error
                }

        except InstructorRetryException as e:
            last_error = f"InstructorRetryException: {str(e)}"
            # 如果是最后一次尝试，必须返回错误，不能 continue
            if retry == max_retries - 1:
                return {
                    "ok": False,
                    "model": None,
                    "completion":e.last_completion,
                    "error": last_error
                }
            continue

        except Exception as e:
            last_error = str(e)
            if retry == max_retries - 1:
                return {
                    "ok": False,
                    "model": None,
                    "completion": None,
                    "error": last_error
                }

    # 【关键】循环意外结束后的兜底返回
    return {
        "ok": False,
        "model": None,
        "completion": None,
        "error": f"Failed after {max_retries} retries. Last error: {last_error}"
    }

# -----------------------------
# 5. 单 prompt 处理
# -----------------------------
async def process_single_prompt_async(
    prompt_item: Dict[str, str],
    model_config: Dict[str, Any],
    pydantic_model: Type[BaseModel],
    client
):
    result = {
        "article_id": prompt_item["article_id"],
        "text": prompt_item["text"],
        "answers": {},
        "resp": "",
    }

    resp = await call_llm_with_retry(
        client=client,
        model=model_config["model_path"],
        prompt=prompt_item["prompt"],
        model_class=pydantic_model,
        temperature=model_config["temperature"],
    )

    if not resp["ok"]:
        result["error"] = resp["error"]
        return result

    model_obj = resp["model"]
    completion = resp["completion"]

    # ✅ 原始 LLM 输出（最重要）
    print("Raw LLM Output:",resp["completion"].choices[0].message.tool_calls[0].function.arguments)
    result["resp"] = (
        completion.choices[0]
        .message
        .tool_calls[0]
        .function
        .arguments
    )


    # ✅ 结构化结果
    result["answers"] = model_obj.model_dump()

    return result


# -----------------------------
# 6. 并发推理
# -----------------------------
async def sent_multi_prompt_async(
    model_config,
    pydantic_model,
    prompt_list,
    client,
    max_concurrency: int = 8
):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(p):
        async with semaphore:
            return await process_single_prompt_async(
                p, model_config, pydantic_model, client
            )

    tasks = [asyncio.create_task(sem_task(p)) for p in prompt_list]
    results = []

    for coro in tqdm_asyncio(asyncio.as_completed(tasks),
                             total=len(tasks),
                             desc="Processing prompts"):
        results.append(await coro)

    return results


# -----------------------------
# 7. 主推理流程
# -----------------------------
@timeit
def run_inference(project_name: str):
    asyncio.run(_run_inference_async(project_name))


async def _run_inference_async(project_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_json(os.path.join(current_dir, "config.json"))["local_inference"]

    client = init_async_client(
        config["model_config"]["base_url"],
        config["model_config"]["api_key"]
    )

    processor = CReDEs_FileProcessor(
        config["data"]["label_mapping_path"],
        config["data"]["CReDEs_mapping_path"]
    )
    cdes = processor.get_CReDEs_mapping_dict()

    pydantic_model = get_pydantic_model(
        cdes, project_name, config["data"]["transfer_data_info"]
    )

    prompts = process_data(project_name, config["data"]["transfer_data_info"])

    results = await sent_multi_prompt_async(
        config["model_config"],
        pydantic_model,
        prompts,
        client,
        max_concurrency=10
    )

    save_data(
        results,
        config["store_transfer_data_path_dir"],
        f"{project_name}_transfer_data.json"
    )


# -----------------------------
# 8. 入口
# -----------------------------
if __name__ == "__main__":
    # run_inference("既往史")
    run_inference("病理")
