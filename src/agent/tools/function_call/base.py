import ast
import operator

from langchain.tools import tool

from src.utils.logger import logger
from src.utils.logger import logger


@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气。"""
    # TODO: 调用真实的天气 API
    logger.info(f"Calling get_weather for city: {city}")
    return f"{city} 的天气是大晴天，25℃。"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。支持基本数学运算：加、减、乘、除、幂运算。"""
    logger.info(f"Evaluating expression: {expression}")

    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
    }

    def safe_eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量类型: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](left, right)
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](operand)
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        else:
            raise ValueError(f"不支持的表达式类型: {ast.dump(node)}")

    try:
        parsed = ast.parse(expression, mode="eval")
        result = safe_eval(parsed.body)
        logger.info(f"Calculation result: {result}")
        return f"表达式 {expression} 的结果是 {result}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        logger.warning(f"Calculation error for expression '{expression}': {e}")
        return f"计算错误: {e}"
    except Exception as e:
        logger.error(f"Unexpected calculation error: {e}")
        return f"计算错误: {e}"
