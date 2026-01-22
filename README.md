# Transfer_entity2CDEs

一个医学信息抽取系统，利用大语言模型（LLM）的显性推理能力，将非结构化的中文医学记录转换为结构化的临床研究数据元素（CDEs）。

## 项目简介

本系统从中文医学记录中提取和验证医学实体，将其转换为标准化的临床研究数据元素（CDEs）。支持多家医院和多种文档类型，具有高级验证和错误处理机制。

### 核心特性

- **显性推理流程**: 四阶段推理过程，确保抽取准确性
- **多医院支持**: 预配置支持东肝（DG）、九院（JY）、胸科（XK）三家医院
- **多种文档类型**: 支持病理诊断、检查、介入治疗、手术记录、日常病程、入院记录等
- **分组实体处理**: 处理复杂的多属性医学实体
- **断点续传**: 支持中断后自动恢复处理
- **异步处理**: 可配置并发数的并发处理
- **数据验证**: 多层验证，包含噪声检测和单位转换

## 目录结构

```
Transfer_entity2CDEs/
├── data/
│   └── Mapping_file/              # 标签和CReDEs映射文件
│       ├── dg/                    # 东肝医院映射文件
│       ├── jy/                    # 九院医院映射文件
│       └── xk/                    # 胸科医院映射文件
├── prompt_info/
│   └── base_pydantic/
│       └── pydantic_model/
│           ├── config.json        # 全局配置文件
│           ├── 3_hosp_config/     # 各医院专用配置
│           │   ├── DG_config.json
│           │   ├── JY_config.json
│           │   └── XK_config.json
│           ├── CReDEsPydanticModel.py  # 核心模型定义
│           ├── params.py          # Prompt模板
│           ├── utils.py           # 工具函数
│           ├── main_async_retry.py      # 异步入口（含显性推理）
│           └── main_retry_completions.py # 同步入口（基础模式）
├── server_sh/
│   └── vllm_server.sh            # VLLM服务器启动脚本
└── requirements_clean.txt        # Python依赖
```

## 安装

### 环境要求

- Python 3.8+
- CUDA兼容的GPU
- Conda环境

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/Xu12875/Transfer_entity2CDEs.git
cd Transfer_entity2CDEs
```

2. 创建并激活conda环境：
```bash
conda create -n LLM-inference python=3.10
conda activate LLM-inference
```

3. 安装依赖：
```bash
pip install -r requirements_clean.txt
```

4. 下载所需模型（如 Qwen3-30B-A3B-Thinking）：
```bash
# 使用 huggingface-cli 或 modelscope
huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking --local-dir /path/to/model
```

## 配置说明

### 配置文件结构

系统使用位于 `prompt_info/base_pydantic/pydantic_model/` 的JSON配置文件。分两级配置：

1. **全局配置** (`config.json`) - 主配置文件
2. **医院专用配置** (`3_hosp_config/*.json`) - 各医院独立配置

### 完整配置示例

```json
{
    "local_inference": {
        "model_config": {
            "model_path": "/data/hf_cache/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
            "base_url": "http://localhost:8085/v1",
            "api_key": "token-abc123",
            "model_name": "/data/hf_cache/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
            "temperature": 0.0,
            "top_p": 0.9,
            "max_token_len": 32768,
            "do_sample": true,
            "top_k": 0,
            "use_beam_search": false,
            "device_map": "4,5,6,7",
            "dtype": "auto"
        },
        "data": {
            "transfer_data_info": {
                "病理诊断": {
                    "entity_inferecne_path": "/path/to/inference_data/blzd.json",
                    "need_grouped_cde_dict": {
                        "肝脏标本": ["切除肝脏总大小横径", "切除肝脏总大小纵径"],
                        "肿瘤大小": ["瘤灶横径", "瘤灶纵径"]
                    }
                },
                "检查": {
                    "entity_inferecne_path": "/path/to/inference_data/jc.json",
                    "need_grouped_cde_dict": {
                        "瘤灶大小": ["瘤灶横径", "瘤灶纵径"]
                    }
                }
            },
            "label_mapping_path": "/path/to/label_mapping.csv",
            "CReDEs_mapping_path": "/path/to/CReDEs_mapping.csv"
        },
        "store_transfer_data_path_dir": "/path/to/output/directory"
    }
}
```

### 配置参数详解

#### Model Config 部分

| 参数 | 类型 | 说明 | 示例 |
|-----------|------|-------------|---------|
| `model_path` | string | HuggingFace模型路径 | `/data/hf_cache/models/Qwen/Qwen3-30B` |
| `base_url` | string | VLLM服务器地址 | `http://localhost:8085/v1` |
| `api_key` | string | API认证密钥 | `token-abc123` |
| `model_name` | string | 模型标识符（通常与model_path相同） | `/data/hf_cache/models/Qwen/Qwen3-30B` |
| `temperature` | float | 采样温度（0.0表示确定性输出） | `0.0` |
| `top_p` | float | 核采样参数 | `0.9` |
| `max_token_len` | int | 最大输出token数 | `32768` |
| `device_map` | string | 用于张量并行的GPU设备ID | `"4,5,6,7"` |
| `do_sample` | bool | 是否使用采样 | `true` |
| `top_k` | int | top-k采样参数 | `0` |
| `dtype` | string | 数据类型 | `"auto"` |

#### Data Config 部分

**`transfer_data_info`**: 定义要处理的文档类型，键名为文档类型名称（如"病理诊断"、"检查"）。

每个文档类型包含：
- `entity_inferecne_path`: 包含待抽取实体的输入JSON文件路径
- `need_grouped_cde_dict`: 定义哪些CDE字段需要分组
  - 键：分组名称（如"肝脏标本"、"肿瘤大小"）
  - 值：属于该分组的字段名列表

**路径配置**:
- `label_mapping_path`: 本地标签到标准标签映射的CSV文件路径
- `CReDEs_mapping_path`: 实体到CReDEs定义映射的CSV文件路径

**输出配置**:
- `store_transfer_data_path_dir`: 结果保存目录

### 如何编写自定义配置

1. **复制模板文件** - 从现有配置文件复制：
```bash
cp prompt_info/base_pydantic/pydantic_model/3_hosp_config/DG_config.json \
   prompt_info/base_pydantic/pydantic_model/my_config.json
```

2. **配置模型设置** - 更新 `model_config` 部分：
   - 设置 `model_path` 为你的模型位置
   - 设置 `base_url` 匹配你的VLLM服务器
   - 根据可用GPU调整 `device_map`
   - 根据模型的上下文窗口设置 `max_token_len`

3. **定义数据路径** - 更新 `data` 部分的路径：
   - 将 `label_mapping_path` 指向你的标签映射CSV
   - 将 `CReDEs_mapping_path` 指向你的CReDEs映射CSV
   - 设置 `store_transfer_data_path_dir` 为你想要的输出目录

4. **配置文档类型** - 在 `transfer_data_info` 中添加/删除条目：
```json
"transfer_data_info": {
    "你的文档类型": {
        "entity_inferecne_path": "/path/to/your/input.json",
        "need_grouped_cde_dict": {
            "分组名称": ["字段1", "字段2", "字段3"]
        }
    }
}
```

5. **设置分组CDEs**（可选）:
   - 当多个字段属于同一实体时使用 `need_grouped_cde_dict`
   - 例如："瘤灶大小" 将 "瘤灶横径" 和 "瘤灶纵径" 分组在一起

### 输入数据格式

输入JSON文件应遵循以下结构：
```json
[
    {
        "article_id": "unique_id_001",
        "text": "原始医学文本内容...",
        "pred_entities_text": "预提取的实体文本..."
    }
]
```

### 映射CSV格式

**标签映射CSV** (`东肝_标签.csv`):
```
标签版本,标签名称,标签类型,属性列表,定义/描述,项目名称
v1,肝脏标本,complex,切除肝脏总大小横径;切除肝脏总大小纵径,肝脏标本描述,病理诊断
```

**CReDEs映射CSV** (`东肝1203_remov_unknown.csv`):
```
CDE名称,数据类型,单位,是否必填,取值范围,...
肝脏标本横径,float,cm,TRUE,0-50,...
肝脏标本纵径,float,cm,TRUE,0-50,...
```

## 使用方法

### 步骤1：启动VLLM服务器

编辑 `server_sh/vllm_server.sh` 以匹配你的GPU设置，然后运行：
```bash
bash server_sh/vllm_server.sh
```

VLLM命令示例：
```bash
vllm serve /path/to/model \
  --host=127.0.0.1 \
  --port=8085 \
  --api-key="token-abc123" \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128
```

### 步骤2：运行抽取程序

**异步处理 + 显性推理模式（推荐）**:
```bash
cd prompt_info/base_pydantic/pydantic_model
python main_async_retry.py
```

**同步处理 + 基础模式**:
```bash
cd prompt_info/base_pydantic/pydantic_model
python main_retry_completions.py
```

### 步骤3：修改项目列表（如需要）

编辑 `main_async_retry.py` 中的 `projects` 列表：
```python
# 东肝医院
projects = ["病理诊断", "检查", "介入治疗", "手术记录", "日常病程", "入院记录"]

# 九院医院
# projects = ["病理检查", "个人史", "婚育史", "既往史", "手术记录", "现病史", "影像检查", "专科检查"]

# 胸科医院
# projects = ["既往史", "家族史", "诊断", "病理", "物理检查", "个人史", "婚育史", "过敏史", "现病史"]
```

## 输出格式

结果保存为JSON文件在配置的输出目录中：

```json
{
    "article_id": "unique_id_001",
    "text": "原始医学文本...",
    "answers": {
        "独立字段名": "提取的值",
        "成组字段名": [
            {
                "属性1": "值1",
                "属性2": "值2"
            }
        ],
        "phase1_non_grouped_analysis": {
            "global_fields": [...],
            "independent_fields": [...]
        },
        "phase2_grouped_analysis": {
            "complex_objects": [...]
        },
        "phase3_synthesis": {
            "conflict_resolutions": [...],
            "consistency_check": "..."
        }
    },
    "error_msg": {},
    "error_type": ""
}
```

## 处理模式

### 基础模式 (`main_retry_completions.py`)
- 直接实体抽取
- 简单验证规则
- 使用 `BAISC_PROMPT`

### 显性推理模式 (`main_async_retry.py`)
- 四阶段推理流程：
  1. **阶段1**：非分组字段分析
  2. **阶段2**：分组实体分析
  3. **阶段3**：综合与冲突解决
  4. **最终阶段**：结构化答案生成
- 使用 `NEW_SYSTEM_PROMPT`
- 更适合复杂的医学实体

## 故障排查

### 服务器连接问题
- 验证VLLM服务器是否运行：`curl http://localhost:8085/v1/models`
- 检查配置中的 `base_url` 和 `api_key`

### GPU显存问题
- 减少VLLM启动时的 `max-model-len`
- 减少 `tensor-parallel-size` 或使用更小的模型
- 降低 `gpu-memory-utilization`

### 断点恢复
系统会自动从中断处恢复。检查 `jsonl_records/` 目录中的临时文件。

## 支持的数据类型

- **文本字段**: 带长度约束的字符串值
- **数值字段**: 带精度和单位的整数/浮点数
- **日期字段**: 格式验证（YYYY-MM-DD等）
- **枚举字段**: 单选/多选预定义选项
- **分组字段**: 具有多个相关属性的复杂对象

## 项目结构说明

```
prompt_info/base_pydantic/pydantic_model/
├── CReDEsPydanticModel.py    # 核心模型定义
│   ├── CReDEs_FileProcessor   # 文件处理器，读取映射文件
│   ├── QARequestFactory       # 动态Pydantic模型工厂
│   ├── MultiQARequestFactory  # 多字段模型工厂
│   └── ReasoningStep/Analysis # 推理模型定义
├── params.py                  # Prompt模板（基础/显性推理）
├── utils.py                   # 工具函数
├── main_async_retry.py        # 异步主程序（显性推理模式）
└── main_retry_completions.py  # 同步主程序（基础模式）
```

## 许可证

MIT License

## 引用

如果您在研究中使用了本代码，请引用：
```bibtex
@software{transfer_entity2cdes,
  title={Transfer_entity2CDEs},
  author={Raince},
  year={2025},
  url={https://github.com/Xu12875/Transfer_entity2CDEs}
}
```
