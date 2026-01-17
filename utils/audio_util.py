import numpy as np
import io
import subprocess

""" 转换音频文件为单声道WAV格式 """
def convert_to_mono(input_file, output_file, sample_rate: int = 16000):
    # ffmpeg command to convert audio to mono wav
    command = [
        'ffmpeg',
        '-i', input_file,  # input file
        '-ac', '1',  # set number of audio channels to 1 (mono)
        '-ar', str(sample_rate),  # set audio sample rate
        '-c:a', 'pcm_s16le',  # set audio codec to PCM 16-bit little-endian
        output_file  # output file
    ]

    # Run ffmpeg command
    # subprocess 是 Python 内置的一个用于调用系统外部命令 / 程序的模块
    # 简单说就是 “让 Python 代码去执行电脑上的其他命令（比如 ffmpeg、cmd 命令等）”
    subprocess.run(command, check=True)

""" 从音频文件读取数据并转换为np.ndarray格式 """
def from_file_to_np_array(input_file: str, dtype: str = "float32") -> (np.ndarray, int):
    import soundfile as sf

    data, samplerate = sf.read(input_file, dtype=dtype)
    return data, samplerate

""" 从字节数据中读取数据并转换为np.ndarray格式 """
def from_bytes_to_np_ndarray(bytes_data: bytes, dtype: str = "float32") -> (np.ndarray, int):
    """
    Convert byte data to np.ndarray format.
    Args:
        bytes_data: Audio bytes of data.
        dtype: Default is float32.

    Returns: Returns the converted np.ndarray format data, sample rate.

    """
    import soundfile as sf

    wave_bytes_buf = io.BytesIO(bytes_data)
    data, samplerate = sf.read(wave_bytes_buf, dtype=dtype)
    return data, samplerate