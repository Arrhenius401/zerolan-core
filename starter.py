import argparse

import yaml

from common.abs_app import AbstractApplication

""""
    这是整个项目的启动入口，通过命令行参数指定要运行的服务类型（如 ASR 语音识别、LLM 大语言模型、OCR 文字识别等）
    脚本会加载配置文件、初始化对应模型和应用，最终启动服务（推测是基于网络的服务，如 HTTP 接口）
"""

""" ==================== 初始化 ==================== """
""" 命令行参数解析 """
parser = argparse.ArgumentParser()
parser.add_argument('service', type=str)  # 必选参数：指定要启动的服务（如asr/llm/ocr）
parser.add_argument('--model', type=str)  # 可选参数：vla服务专用，指定模型类型
parser.add_argument('--db', type=str)     # 可选参数：vecdb服务专用，指定数据库类型
parser.add_argument('--config', type=str) # 可选参数：指定配置文件路径（默认./config.yaml）
args = parser.parse_args()

""" 加载配置文件 """
def load_config():
    path = args.config if args.config else './config.yaml'
    with open(path, mode='r', encoding='utf-8') as f:
        return yaml.safe_load(f)


_config = load_config()


""" ==================== 核心功能模块：各服务的初始化函数 ==================== """
""" ASR 服务（语音识别） """
def asr_app() -> AbstractApplication:
    from asr.app import ASRApplication

    asr_config = _config["ASR"]         # 读取ASR整体配置
    asr_id = asr_config["id"]           # 读取要使用的模型ID
    model_cfg = asr_config["config"][asr_id]  # 读取该模型的具体配置

    def get_model():
        print(asr_id)
        # 根据模型ID分支加载不同模型
        if asr_id == "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1":
            from asr.paraformer.model import SpeechParaformerModel as Model
            from asr.paraformer.config import SpeechParaformerModelConfig as Config
            # ** 是 Python 里的关键字参数解包运算符（也叫字典解包）
            # 核心作用是把字典类型的 model_cfg 中的键值对，自动拆解成 Config 类初始化时需要的关键字参数
            return Model(Config(**model_cfg))
        elif asr_id == "kotoba-tech/kotoba-whisper-v2.0":
            from asr.kotoba_whisper_2.model import KotobaWhisper2 as Model
            from asr.kotoba_whisper_2.config import KotobaWhisper2Config as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {asr_id}")

    asr = get_model()
    app = ASRApplication(
        model=asr, host=asr_config["host"], port=asr_config["port"])
    return app

""" LLM 服务（大语言模型） """
def llm_app() -> AbstractApplication:
    from llm.app import LLMApplication

    llm_config = _config["LLM"]
    llm_id = llm_config["id"]
    model_cfg = llm_config["config"][llm_id]

    def get_model():
        if llm_id == "THUDM/chatglm3-6b":
            from llm.chatglm3.model import ChatGLM3_6B as Model
            from llm.chatglm3.config import ChatGLM3ModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id == "Qwen/Qwen-7B-Chat":
            from llm.qwen.model import Qwen7BChat as Model
            from llm.qwen.config import QwenModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id == "augmxnt/shisa-7b-v1":
            from llm.shisa.model import Shisa7B_V1 as Model
            from llm.shisa.config import ShisaModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id == "01-ai/Yi-6B-Chat":
            from llm.yi.model import Yi6B_Chat as Model
            from llm.yi.config import YiModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id == "THUDM/glm-4-9b-chat-hf":
            from llm.glm4.model import GLM4_9B_Chat_Hf as Model
            from llm.glm4.config import GLM4ModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id in ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"]:
            from llm.deepseek.model import DeepSeekLLMModel as Model
            from llm.deepseek.config import DeepSeekModelConfig as Config
            return Model(Config(**model_cfg))
        elif llm_id == "ollama":
            from llm.ollama.model import OllamaModel as Model
            from llm.ollama.config import OllamaModelConfig as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {llm_id}")

    llm = get_model()
    app = LLMApplication(
        model=llm, host=llm_config["host"], port=llm_config["port"])
    return app

""" ImgCap 服务（图像描述生成） """
def imgcap_app() -> AbstractApplication:
    from img_cap.app import ImgCapApplication

    imgcap_config = _config["ImgCap"]
    imgcap_id = imgcap_config["id"]
    model_cfg = imgcap_config["config"][imgcap_id]

    def get_model():
        if imgcap_id == "Salesforce/blip-image-captioning-large":
            from img_cap.blip.model import BlipImageCaptioningLarge as Model
            from img_cap.blip.config import BlipModelConfig as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {imgcap_id}")

    imgcap = get_model()
    app = ImgCapApplication(
        model=imgcap, host=imgcap_config["host"], port=imgcap_config["port"])
    return app

""" OCR 服务（文字识别） """
def ocr_app() -> AbstractApplication:
    from ocr.app import OCRApplication

    ocr_config = _config["OCR"]
    ocr_id = ocr_config['id']
    model_cfg = ocr_config["config"][ocr_id]

    def get_model():
        if ocr_id == "paddlepaddle/PaddleOCR":
            from ocr.paddle.model import PaddleOCRModel as Model
            from ocr.paddle.config import PaddleOCRModelConfig as Config
            return Model(Config(**model_cfg))
        else:
            raise NameError(f"No such model name (id) {ocr_id}")

    ocr = get_model()
    app = OCRApplication(
        model=ocr, host=ocr_config["host"], port=ocr_config["port"])
    return app

""" TTS 服务（文本转语音） """
def tts_app() -> AbstractApplication:
    from tts.app import TTSApplication

    tts_config = _config["TTS"]
    tts_id = tts_config["id"]
    model_cfg = tts_config["config"][tts_id]

    def get_model():
        if tts_id == "AkagawaTsurunaki/GPT-SoVITS":
            from tts.gpt_sovits.model import GPT_SoVITS as Model
            return Model()

    tts = get_model()
    app = TTSApplication(
        model=tts, host=tts_config["host"], port=tts_config["port"])
    return app

""" VLA 服务（视觉语言分析） """
def vla_app(model) -> AbstractApplication:
    if "showui" in model:
        config = _config["VLA"]["ShowUI"]
        model_cfg = config["config"]
        from vla.showui.app import ShowUIApplication as App
        from vla.showui.config import ShowUIModelConfig as Config
        from vla.showui.model import ShowUIModel as Model

        model = Model(Config(**model_cfg))
        app = App(model=model, host=config["host"], port=config["port"])
        return app

""" VecDB 服务（向量数据库） """
def vecdb_app(db):
    if "milvus" in db:
        config = _config["database"]["milvus"]
        from database.milvus.milvus import MilvusApplication as App, MilvusDatabase as DB
        from database.milvus.config import MilvusDBConfig as Config
        db_config = config["config"]

        config = Config(**db_config)
        database = DB(config)
        app = App(database=database, host=config.host, port=config.port)
        return app




""" ==================== 服务路由与启动 ==================== """
""" 
    1. 服务路由：get_app(service)
    根据传入的service参数，分发到对应服务的初始化函数：
"""
def vidcap_app():
    config = _config["VidCap"]
    host, port = config["host"], config["port"]
    model_id = config["id"]

    from vid_cap.app import VidCapApplication
    from vid_cap.hitea.model import HiteaBaseModel
    from vid_cap.hitea.config import HiteaBaseModelConfig

    config = HiteaBaseModelConfig(**config['config'][model_id])
    model = HiteaBaseModel(config)
    app = VidCapApplication(model=model, host=host, port=port)
    return app

""" ==================== 服务路由与启动 ==================== """
""" 
    1. 服务路由：get_app(service)
    根据传入的service参数，分发到对应服务的初始化函数：
"""
def get_app(service):
    if "asr" == service:
        return asr_app()
    elif "llm" == service:
        return llm_app()
    elif "imgcap" == service:
        return imgcap_app()
    elif "ocr" == service:
        return ocr_app()
    elif "tts" == service:
        return tts_app()
    elif "vla" == service:
        return vla_app(args.model)
    elif "vecdb" == service:
        return vecdb_app(args.db)
    elif "vidcap" == service:
        return vidcap_app()
    else:
        raise NotImplementedError("Unsupported service.")

"""
    2. 启动入口：run(service)
"""
def run(service=None):
    service = args.service if service is None else service
    print(service)
    app = get_app(service)
    app.run()


run()   # 执行启动
