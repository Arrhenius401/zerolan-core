"""
More details about the model:
    https://github.com/THUDM/ChatGLM3
"""
from loguru import logger
from transformers import AutoTokenizer, AutoModel

from common.abs_model import AbstractModel
from common.decorator import log_model_loading
from llm.chatglm3.config import ChatGLM3ModelConfig
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction, Conversation


class ChatGLM3_6B(AbstractModel):

    def __init__(self, config: ChatGLM3ModelConfig):
        super().__init__()
        self.model_id = "THUDM/ChatGLM3"
        self._model_path = config.model_path
        self._quantize = config.quantize
        self._device = config.device

        self._tokenizer: any = None
        self._model: any = None

    """ 加载模型 """
    @log_model_loading("THUDM/ChatGLM3")
    def load_model(self):
        # 加载tokenizer：本地无则从网络下载
        # self._model_path 对应配置文件中的模型路径（如 THUDM/chatglm3-6b），若本地路径无该模型，会自动从 Hugging Face Hub 下载 tokenizer 配置和词汇表。
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        if self._quantize:
            self._model = AutoModel.from_pretrained(self._model_path, trust_remote_code=True).quantize(
                self._quantize).to(
                self._device).eval()
            logger.info(f"Model is loaded as {self._quantize}")
        else:
            self._model = AutoModel.from_pretrained(self._model_path, trust_remote_code=True).to(self._device).eval()
            logger.info(f"Model is loaded without quantization.")
        assert self._tokenizer and self._model

    """ 推理方法：完成输入解析 → 模型推理 → 结果格式化 """
    def predict(self, llm_query: LLMQuery) -> LLMPrediction:
        """
        Predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan.data.pipeline.llm.LLMQuery

        Returns: See zerolan.data.pipeline.llm.LLMPrediction

        """
        # 1. 格式转换：将通用的LLMQuery转为ChatGLM3要求的输入格式
        text, history = self._to_chatglm_format(llm_query)
        # Note: In the new version, past_key_values=None throws IndexError,
        # Because the underlying code does not determine whether past_key_values is None or not,
        # Instead, try to parse as long as there is a past_key_values parameter
        # 2. 模型推理：调用ChatGLM3的chat接口，完成文本生成（内部会处理token）
        response, history = self._model.chat(self._tokenizer, text, history, top_p=1., temperature=1.)
        logger.debug(response)
        # 3. 结果格式化：将ChatGLM3的输出转为通用的LLMPrediction格式
        return self._to_pipeline_format(response, history)

    """ 流式推理方法：完成输入解析 → 模型流式推理 → 结果格式化 """
    def stream_predict(self, llm_query: LLMQuery):
        """
        Stream predict tokens based on history and query from LLM.
        Args:
            llm_query: See zerolan.data.pipeline.llm.LLMQuery

        Returns: See zerolan.data.pipeline.llm.LLMPrediction

        """
        text, history = self._to_chatglm_format(llm_query)
        for response, history, past_key_values in self._model.stream_chat(self._tokenizer, text, history=history,
                                                                          top_p=1.,
                                                                          temperature=1.,
                                                                          past_key_values=None,
                                                                          return_past_key_values=True):
            logger.debug(response)
            yield self._to_pipeline_format(response, history)

    """ 将通用的LLMQuery格式转为ChatGLM3要求的输入格式 """
    @staticmethod
    def _to_chatglm_format(llm_query: LLMQuery) -> (str, list[dict[str:str]]):
        text = llm_query.text
        history = [{'role': chat.role, 'metadata': '', 'content': chat.content} for chat in llm_query.history]
        return text, history

    """ 格式化模型输出为通用的LLMPrediction格式 """
    @staticmethod
    def _to_pipeline_format(response: str, history: list[dict[str:str]]) -> LLMPrediction:
        history = [Conversation(role=chat['role'], content=chat['content']) for chat in history]
        llm_response = LLMPrediction(response=response, history=history)
        return llm_response
