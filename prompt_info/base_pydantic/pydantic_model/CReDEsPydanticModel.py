import re
import pandas as pd
from typing import List, Dict, Union, Optional, Any, Type, Annotated,Literal
from typing_extensions import TypeVar
from pydantic import BaseModel, Field, create_model, AfterValidator, ConfigDict, BeforeValidator
from datetime import datetime, date

# 定义类型变量用于运行时类型注解
NumberType = TypeVar('NumberType', int, float)

class CReDEs_FileProcessor:
    def __init__(self, label_csv_file_path: str, CReDEs_file_path: str): # 修正了 pathg -> path
        self.label_csv_file_path = label_csv_file_path
        self.CReDEs_file_path = CReDEs_file_path

    def _get_label_mapping_list(self) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.label_csv_file_path)
        label_item_list = []
        for _, row in df.iterrows():
            label_id = str(row['标签版本']).strip()
            label_name = str(row['标签名称']).strip()
            label_desc = str(row['定义/描述']).strip()
            label_type = str(row['标签类型']).strip()
            label_attr_str = str(row.get('属性列表', '')).strip()
            label_attr_list = []

            if label_attr_str:
                label_attr = [x.strip() for x in label_attr_str.split(';') if x.strip()]
                for attr in label_attr:
                    if ':' in attr:
                        attr_name = attr.split(':')[0]
                        label_attr_list.append(attr_name)

            project_name = str(row.get('项目名称', '')).strip()
            label_temp_item = {
                "label_id": label_id,
                "label_name": label_name,
                "label_desc": label_desc,
                "label_type": label_type,
                "label_attr": label_attr_list,
                "project_name": project_name
            }
            label_item_list.append(label_temp_item)
        return label_item_list

    def _get_group_label_mapping_list(self) -> Dict[str, List[Dict[str, Any]]]:
        # Step 1: 按项目分组
        label_item_list = self._get_label_mapping_list()
        project_label_dict: Dict[str, List[Dict[str, Any]]] = {}
        for label_item in label_item_list:
            project_name = label_item['project_name']
            project_label_dict.setdefault(project_name, []).append(label_item)

        # Step 2: 构建分组
        project_group_label_dict: Dict[str, List[Dict[str, Any]]] = {}

        for project_name, labels in project_label_dict.items():
            # 建立 name→label 索引表
            name_to_label = {item["label_name"]: item for item in labels}
            grouped_labels = []

            for label_item in labels:
                # 仅处理实体类型
                if label_item["label_type"] != "实体标签":
                    continue

                label_name = label_item["label_name"]
                label_attr = label_item["label_attr"]

                group_dict = {
                    label_name: {
                        "label_id": label_item["label_id"],
                        "label_desc": label_item["label_desc"]
                    }
                }
                # 找到对应的属性标签
                for attr_name in label_attr:
                    attr_item = name_to_label.get(attr_name)
                    if attr_item and attr_item["label_type"] == "属性标签":
                        group_dict[attr_name] = {
                            "label_id": attr_item["label_id"],
                            "label_desc": attr_item["label_desc"]
                        }

                grouped_labels.append(group_dict)

            project_group_label_dict[project_name] = grouped_labels

        return project_group_label_dict
    
    def _safe_int(self, v: Any) -> Union[int, str]:
        try:
            v_str = str(v).strip()
            if v_str == "" or v_str.lower() == "nan":
                return ""
            return int(float(v_str))  # 有时数字可能是小数，比如"3.0"
        except (ValueError, TypeError):
            return ""

    def _get_CReDEs_basic_info(self) -> List[Dict[str, Dict[str, str]]]:
        df = pd.read_csv(self.CReDEs_file_path)
        df = df.fillna("")

        CReDEs_list = []
        for _, row in df.iterrows():
            label_id = str(row.get('标签版本', '')).strip()
            CReDEs_metadata_name = str(row.get('CReDEs-数据项', '')).strip()
            CReDEs_metadata_dataType = str(row.get('数据类型', '')).strip()
            CReDEs_metadata_textLength = self._safe_int(row.get('文本-长度', ''))
            CReDEs_metadata_numLength = self._safe_int(row.get('数字-长度', ''))
            CReDEs_metadata_numPrecision = self._safe_int(row.get('数字-精度', ''))
            CReDEs_metadata_numUnit = str(row.get('数字-单位', '')).strip()
            CReDEs_metadata_dateFormat = str(row.get('日期/时间', '')).strip()
            CReDEs_metadata_enumInfo = str(row.get('枚举-可选内容', '')).strip()
            CReDEs_metadata_enumUnique = str(row.get('枚举-是否多选', '')).strip()

            CReDEs_info_dict = {
                "label_id": label_id,
                "CReDEs_metadata_name": CReDEs_metadata_name,
                "CReDEs_metadata_dataType": CReDEs_metadata_dataType,
                "CReDEs_metadata_textLength": CReDEs_metadata_textLength,
                "CReDEs_metadata_numLength": CReDEs_metadata_numLength,
                "CReDEs_metadata_numPrecision": CReDEs_metadata_numPrecision,
                "CReDEs_metadata_numUnit": CReDEs_metadata_numUnit,
                "CReDEs_metadata_dateFormat": CReDEs_metadata_dateFormat,
                "CReDEs_metadata_enumInfo": CReDEs_metadata_enumInfo,
                "CReDEs_metadata_enumUnique": CReDEs_metadata_enumUnique
            }
            if CReDEs_info_dict['CReDEs_metadata_enumInfo'] is not None and CReDEs_info_dict['CReDEs_metadata_enumInfo'] != "":
                CReDEs_metadata_dataType = "枚举"
                CReDEs_info_dict['CReDEs_metadata_dataType'] = CReDEs_metadata_dataType
            CReDEs_list.append(CReDEs_info_dict)

        return CReDEs_list
    

    def process_cobime_CReDEs_metadata_item(self, item_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并同名 CReDEs metadata：
        - item 完全相同 → 去重
        - item 不完全相同 → 
            label_name  = 标签拼接
            label_desc  = "label_name：label_desc" 拼接
        - 其他字段保持不变（基于第一个 item）
        """

        # 1. 去重
        unique_items = []
        seen = set()
        for item in item_list:
            key = tuple(sorted(item.items()))
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        # 如果完全相同，直接返回
        if len(unique_items) == 1:
            return unique_items[0]

        # 2. 存在不同，需要合并
        base_item = unique_items[0].copy()

        label_names = []
        desc_pairs = []

        for item in unique_items:
            ln = item.get("label_name", "")
            ld = item.get("label_desc", "")

            if ln:
                label_names.append(ln)

            if ln or ld:
                desc_pairs.append(f"{ln}：{ld}")

        # 去重
        label_names = sorted(set(label_names))
        desc_pairs = sorted(set(desc_pairs))

        base_item["label_name"] = "；".join(label_names)          # 只标签
        base_item["label_desc"] = "；".join(desc_pairs)          # label：desc 列表

        return base_item

    def combine_same_CReDEs_metadata(self, CReDEs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并相同标签ID的CReDEs元数据
        """
        combined_CReDEs_list = []

        # 建立唯一名称列表：CReDEs_metadata_name 优先，没有就 label_name
        cde_name_unique_list = list({
            item.get('CReDEs_metadata_name') or item.get('label_name')
            for item in CReDEs_list
            if item.get('CReDEs_metadata_name') or item.get('label_name')
        })

        for cde_name in cde_name_unique_list:
            cde_combine_item_list = []
            for item in CReDEs_list:
                # ★ fallback，一致就归类
                item_name = item.get('CReDEs_metadata_name') or item.get('label_name')
                if item_name == cde_name:
                    cde_combine_item_list.append(item)

            if not cde_combine_item_list:
                # 理论不应该发生，加个保护避免崩溃
                continue

            if len(cde_combine_item_list) > 1:
                base = self.process_cobime_CReDEs_metadata_item(cde_combine_item_list)
                combined_CReDEs_list.append(base)
            else:
                combined_CReDEs_list.append(cde_combine_item_list[0])

        return combined_CReDEs_list
            

    def get_CReDEs_mapping_dict(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        获取最终映射字典，结构为: {项目名: {实体标签名: [合并后的信息列表]}}
        """
        project_group_label_dict = self._get_group_label_mapping_list()
        CReDEs_list = self._get_CReDEs_basic_info()

        # 构建索引表：label_id -> 所有对应的CReDEs元数据（可能一对多）
        CReDEs_index: Dict[str, List[Dict[str, Any]]] = {}
        for item in CReDEs_list:
            CReDEs_index.setdefault(item['label_id'], []).append(item)

        # 最终返回的字典
        final_CReDEs_dict: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for project_name, group_label_item_list in project_group_label_dict.items():
            # 每个项目对应一个字典，key是实体标签名
            project_name = str(project_name).split("-")[-1].strip()
            # print(f"Processing project: {project_name}")
            project_entity_dict: Dict[str, List[Dict[str, Any]]] = {}

            for group_label_item in group_label_item_list:
                # group_label_item 是一个字典，如: {'实体A': {...}, '属性B': {...}}
                # 我们需要找到实体标签名作为key
                entity_name = next(iter(group_label_item)) # 获取字典的第一个键，即实体名
                
                # 该实体及其所有属性对应的合并信息列表
                merged_info_list: List[Dict[str, Any]] = []

                for label_name, label_item in group_label_item.items():
                    label_id = label_item['label_id']
                    related_metadata_list = CReDEs_index.get(label_id, [])

                    # 如果该标签有对应的多个 CReDEs 条目，全都加入
                    if related_metadata_list:
                        for meta in related_metadata_list:
                            merged_info = {
                                "label_name": label_name,
                                "label_id": label_id,
                                "label_desc": label_item.get("label_desc", "")
                            }
                            merged_info.update(meta)
                            merged_info_list.append(merged_info)
                    else:
                        # 即使没有匹配的CReDEs，也保留label信息
                        merged_info_list.append({
                            "label_name": label_name,
                            "label_id": label_id,
                            "label_desc": label_item.get("label_desc", "")
                        })
                
                # 将构建好的列表赋值给对应的实体标签名
                project_entity_dict[entity_name] = merged_info_list
            
            # 将每个项目的字典放入最终结果中
            # final_CReDEs_dict[project_name] = project_entity_dict
            final_CReDEs_dict[project_name] = [item for v in project_entity_dict.values() for item in v]
            
        # 合并相同名的CReDEs元数据
        for project_name, cde_list in final_CReDEs_dict.items():
            combined_cde_list = self.combine_same_CReDEs_metadata(cde_list)
            final_CReDEs_dict[project_name] = combined_cde_list

        return final_CReDEs_dict

    
class QARequestFactory:
    """QA请求模型工厂类（Pydantic v2 最佳实践修正版）"""

    @staticmethod
    def create_field_definition(label_data: Dict[str, Any]) -> tuple:
        """
        创建字段定义元组 (type, field_info)，用于后续模型创建
        返回: (field_name, field_type, field_info)
        """
        try:
            metadata_type = label_data.get("CReDEs_metadata_dataType", "文本")
        except:
            metadata_type = "文本"

        if metadata_type == "枚举":
            return QARequestFactory._create_enum_field(label_data)
        elif metadata_type in ["整数", "小数"]:
            target_type = int if metadata_type == "整数" else float
            return QARequestFactory._create_number_field(label_data, target_type)
        elif metadata_type == "日期/时间":
            return QARequestFactory._create_date_field(label_data)
        else:
            return QARequestFactory._create_text_field(label_data)



    # -------------------------------------------------------------------------
    # 通用验证逻辑函数 (用于 AfterValidator)
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _validate_or_fallback(v: Any, validator_func, fallback_str="") -> Any:
        """通用验证器包装：允许特定字符串通过，否则执行验证逻辑"""
        if isinstance(v, str) and v.strip() == fallback_str:
            return v
        return validator_func(v)

    # -------------------------------------------------------------------------
    # 模型构建方法
    # -------------------------------------------------------------------------

    @staticmethod
    def _create_enum_field(label_data: Dict[str, Any]) -> tuple:
        label_name = label_data.get("label_name", "")
        label_desc = label_data.get("label_desc", "")
        metadata_name = label_data.get("CReDEs_metadata_name", "") or label_name

        enum_info = label_data.get("CReDEs_metadata_enumInfo", "")
        enum_unique = label_data.get("CReDEs_metadata_enumUnique", "否")

        # 解析：{"1":"无","2":"男","3":"女"}
        options_map = QARequestFactory._parse_enum_options(enum_info)
        allowed_values = list(options_map.values())  # 只允许文本值
        if "" not in allowed_values:
            allowed_values.append("")

        def validate_enum_value(v: Any) -> str:
            v_str = str(v).strip()
            if v_str in allowed_values:
                return v_str
            # raise ValueError(
            #     f"值 '{v}' 不在允许的选项中。可选值包括：{', '.join(allowed_values)}"
            # )

        # 选项文本格式（不包含编号）
        enum_values_str = "、".join(options_map.values())

        # 字段描述
        description_str = (
            f"数据来源于 {label_name}。描述：{label_desc}。"
            f"可选值包括：{enum_values_str}。"
            f"若文中没有描述{metadata_name}，填 ''。"
        )

        # 构造字段类型
        DynamicLiteral = Literal[tuple(allowed_values)]
        DynamicEnumField = Annotated[
            DynamicLiteral, 
            BeforeValidator(lambda v: QARequestFactory._validate_or_fallback(v, validate_enum_value)),
            Field(description=description_str)
        ]

        is_multi = QARequestFactory._True_false_to_bool(enum_unique)

        # 多选 / 单选
        if is_multi:
            # 如果是多选，就是 List[Literal["A", "B"]]
            field_type = List[DynamicEnumField]
            field_info = Field(..., description=description_str + "（可多选）")
        else:
            # 如果是单选，就是 Literal["A", "B"]
            field_type = DynamicEnumField
            field_info = Field(..., description=description_str + "（单选）")

        return metadata_name, field_type, field_info
        

    @staticmethod
    def _create_number_field(label_data: Dict[str, Any], target_type: Union[Type[int], Type[float]]) -> tuple:
        label_name = label_data.get("label_name", "")
        label_desc = label_data.get("label_desc", "")
        metadata_name = label_data.get("CReDEs_metadata_name", "")
        if metadata_name is None or metadata_name == "":
            metadata_name = label_name
        num_length = label_data.get("CReDEs_metadata_numLength") # 整数总长度
        num_precision = label_data.get("CReDEs_metadata_numPrecision") # 小数位
        num_unit = label_data.get("CReDEs_metadata_numUnit", "")

        # 构造验证函数
        def validate_number(v: Any) -> Optional[NumberType]:
            # 1. 特殊文本情况直接返回 None
            if isinstance(v, str):
                # 医学中常见的非数值描述情况
                non_numeric_keywords = ["多发", "数个", "多个", "未见", "不详", "无法", "未测量", "未描述", ""]

                # 如果完全匹配"无相关描述"或空字符串
                if not v.strip() or v in non_numeric_keywords:
                    return None

                # 尝试将字符串中的数字部分解析
                cleaned = v.replace("约", "").replace("cm", "").replace("个", "").strip()
                try:
                    return target_type(cleaned)
                except Exception:
                    # 解析失败 → 返回 None
                    return None

            # 2. 如果本身是数字或可直接转数字
            try:
                val = target_type(v)
            except Exception:
                # 不是数字，返回 None
                return None

            # 3. 数值长度检查
            if num_length:
                str_val = str(val).replace('.', '').replace('-', '')
                if len(str_val) > int(num_length):
                    # 长度超过则返回 None
                    return None
            return val

        desc = f"数据主要来源于 {label_name} 标签。描述：{label_desc}。请填写{target_type.__name__}。"
        if num_unit: desc += f" 单位：{num_unit}。"
        desc += f" 若文中没有对此{metadata_name}的描述，填null。"

        # 定义字段类型：允许目标数值类型或 None
        field_type = Annotated[
            Union[NumberType, None],
            AfterValidator(validate_number),  # 直接使用验证函数，不需要 _validate_or_fallback
            Field(description=desc)
        ]

        field_info = Field(default=None, description=desc)

        return metadata_name, field_type, field_info

    @staticmethod
    def _create_date_field(label_data: Dict[str, Any]) -> tuple:
        label_name = label_data.get("label_name", "")
        label_desc = label_data.get("label_desc", "")
        metadata_name = label_data.get("CReDEs_metadata_name", "")
        if metadata_name is None or metadata_name == "":
            metadata_name = label_name

        input_format = label_data.get("CReDEs_metadata_dateFormat", "YYYY-MM-DD")

        py_format = input_format.replace("YYYY", "%Y").replace("MM", "%m").replace("DD", "%d")

        def validate_date(v: Any) -> Any:
            if v == "":
                return v
            if isinstance(v, (date, datetime)):
                return v.date() if isinstance(v, datetime) else v

            try:
                return datetime.strptime(str(v), py_format).date()
            except:
                return v

        desc = f"数据主要来源于 {label_name} 标签。描述：{label_desc}。若文中没有对此{metadata_name}的描述，填''。"

        field_type = Annotated[
            Union[date, str], # 允许 date 对象或字符串输入
            # AfterValidator(validate_date),
            AfterValidator(lambda v: QARequestFactory._validate_or_fallback(v, validate_date)),
            Field(description=desc)
        ]

        field_info = Field(..., description=desc)

        return metadata_name, field_type, field_info

    @staticmethod
    def _create_text_field(label_data: Dict[str, Any]) -> tuple:
        label_name = label_data.get("label_name", "")
        label_desc = label_data.get("label_desc", "")
        metadata_name = label_data.get("CReDEs_metadata_name", "")
        if metadata_name is None or metadata_name == "":
            metadata_name = label_name
        text_length = QARequestFactory._get_int_config(label_data, "CReDEs_metadata_textLength", 500)

        desc = f"数据主要来源于 {label_name} 标签。描述：{label_desc}。请填写文本。若文中没有对此{metadata_name}的描述，填''。"

        # 文本模型相对简单，直接使用 Field 的 max_length
        # 但如果要支持严格的 '无相关描述' 检查，也可以加 validator
        if metadata_name is None or metadata_name == "":
            metadata_name = label_name

        field_type = str
        field_info = Field(..., max_length=text_length, description=desc)

        return metadata_name, field_type, field_info
    @staticmethod
    def _parse_enum_options(enum_info: str) -> Dict[str, str]:
        """解析枚举字符串 '1=男; 2=女' -> {'1': '男', '2': '女'}"""
        options = {}
        if not enum_info:
            return options

        # 标准化分隔符
        items = re.sub(r"[；;]", ";", enum_info).split(";")

        auto_code = 1
        for item in items:
            item = item.strip()
            if not item:
                continue

            # 情况 1：包含 "="，按 code=value 解析
            if "=" in item:
                code, desc = item.split("=", 1)
                code = code.strip()
                desc = desc.strip()
                if code and desc:
                    options[code] = desc
            else:
                # 情况 2：无 "=", 纯文本 -> 自动编号
                options[str(auto_code)] = item
                auto_code += 1

        return options

    @staticmethod
    def _True_false_to_bool(value: Any) -> bool:
        """宽松的布尔值转换"""
        if isinstance(value, bool): 
            return value
        str_val = str(value).lower().strip()
        true_values = {"是", "有", "存在", "positive", "true", "1"}
        if str_val in true_values:
            return True
        return False
    @staticmethod
    def _get_int_config(data: Dict[str, Any], key: str, default: int = None) -> Union[int, None]:
        """安全获取配置项，处理空字符串的情况"""
        val = data.get(key)
        if val is None or str(val).strip() == "":
            return default


class MultiQARequestFactory:
    """多QA请求模型工厂类（已优化以使用新的统一模型创建方法）"""
    @staticmethod
    def create_multi_qa_model(label_data_list: List[Dict[str, Any]], model_name: str = "DynamicMultiQARequest") -> Type[BaseModel]:
        """
        创建包含多个字段的统一QA请求模型
        这样可以避免为每个字段都创建一个独立的模型类
        """
        field_definitions = {}

        for label_data in label_data_list:
            field_name, field_type, field_info = QARequestFactory.create_field_definition(label_data)
            field_definitions[field_name] = (field_type, field_info)
        
        return create_model(model_name, **field_definitions)


#  --- 单步推理 ---
class ReasoningStep(BaseModel):
    """单步推理原子"""
    subquestion: str = Field(..., description="当前关注的特定字段或问题。")
    source_evidence: str = Field(..., description="原文中对应的具体描述片段（直接引用）。")
    analysis: str = Field(..., description="分析逻辑：对比标注与原文，判断真伪，处理否定词。")
    conclusion: str = Field(..., description="该步骤得出的当前CDE的结果（尚未填入最终字段）。")

# --- 第一部分：非成组分析 ---
class NonGroupedAnalysis(BaseModel):
    """Step 1: 全局与独立字段分析"""
    overall_context: str = Field(..., description="对病历类型的整体判断（如：这是一份入院记录还是手术记录？）。")
    steps: List[ReasoningStep] = Field(..., description="针对每一个非成组CDE字段逐个推理。")

# --- 第二部分：成组分析 ---
class GroupedAnalysis(BaseModel):
    """Step 2: 复杂对象与成组字段分析"""
    identification: str = Field(..., description="先分析原文和标注数据中包含几个目标实体（如：发现了3个肿瘤，或2次手术），如发现标注数据与原文冲突，则按原文进行分析。")
    group_steps: List[ReasoningStep] = Field(..., description="针对每个实体内部属性的结合原文进行，详细推理（如：肿瘤1的大小，肿瘤2的位置）。")

# --- 第三部分：整合与校验 ---
class SynthesisAnalysis(BaseModel):
    """Step 3: 最终整合与逻辑自洽性检查"""
    conflict_resolution: str = Field(..., description="如果在上述两步中发现原文与标注有冲突，在此总结最终采信的依据。")
    completeness_check: str = Field(..., description="检查是否遗漏了原文中的重要阴性信息（如'否认xxx'）。")
    final_instruction: str = Field(..., description="给填充动作的最终指令（例如：'确认数组不为空，数值已换算'）。")