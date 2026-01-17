import os.path

from flask import Flask, request, jsonify, Response
from loguru import logger

from common.abs_app import AbstractApplication
from utils import audio_util, file_util, web_util
from zerolan.data.pipeline.asr import ASRQuery, ASRStreamQuery, ASRPrediction


class ASRApplication(AbstractApplication):
    def __init__(self, model, host: str, port: int):
        # Python 中调用父类的 __init__ 方法时，super().__init__(...) 里不需要显式传递 self 参数
        # super() 的本质：自动绑定当前实例（self）
        super().__init__(model, "asr")
        self.host = host
        self.port = port
        self._app = Flask(__name__)

    def run(self):
        self.model.load_model()
        self.init()
        self._app.run(self.host, self.port, False)

    def init(self):
        @self._app.route('/asr/predict', methods=['POST'])
        def handle_predict():
            logger.info('Request received: processing...')

            query: ASRQuery = web_util.get_obj_from_json(request, ASRQuery)

            # os.path.exists(path)：用于判断指定路径（文件或目录）是否存在于当前文件系统中
            # 若路径存在（无论它是文件、文件夹，甚至是符号链接指向的有效路径）→ 返回 True；
            # 若路径不存在、路径无效（如含非法字符）、权限不足无法访问 → 返回 False。
            if not os.path.exists(query.audio_path)  or not os.path.isfile(query.audio_path):
                # 路径为空 或 不是文件（是目录/无效文件），保存请求中的音频
                logger.warning("Audio file not found. Saving audio file...")
                audio_path = web_util.save_request_audio(request, prefix="asr")
            else:
                logger.debug(f"Audio file found at: {query.audio_path}")
                audio_path = query.audio_path

            # Convert to mono channel audio file.
            # Warning: Using ffmpeg for conversion can create performance issues
            if query.channels != 1:
                mono_audio_path = file_util.create_temp_file(prefix="asr", suffix=".wav", tmpdir="audio")
                audio_util.convert_to_mono(audio_path, mono_audio_path, query.sample_rate)
                query.audio_path = mono_audio_path
            else:
                # Fixed: Or it will load original file path (for example local machine)
                query.audio_path = audio_path

            prediction: ASRPrediction = self.model.predict(query)
            logger.info(f"Response: {prediction.transcript}")
            return Response(
                response=prediction.model_dump_json(),
                status=200,
                mimetype='application/json',
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )

        @self._app.route('/asr/stream-predict', methods=['POST'])
        def handle_stream_predict():
            query: ASRStreamQuery = web_util.get_obj_from_json(request, ASRStreamQuery)
            audio_data = web_util.get_request_audio_file(request).stream.read()
            query.audio_data = audio_data

            prediction: ASRPrediction = self.model.stream_predict(query)
            return jsonify(prediction.model_dump())
