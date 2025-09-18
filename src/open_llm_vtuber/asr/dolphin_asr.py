from .asr_interface import ASRInterface
import dolphin  # 假设你已安装 dolphin 包
import numpy as np

class VoiceRecognition(ASRInterface):
    def __init__(self, model_path=None, device="cuda", **kwargs):
        # 这里初始化 Dolphin 模型
        self.model = dolphin.load_model("small", model_path, device)

    async def async_transcribe_np(self, audio: np.ndarray) -> str:
        # Dolphin 需要文件路径，所以需要保存临时文件
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, 16000)  # 假设采样率为16k
            result = self.model(dolphin.load_audio(tmp.name))
            # print(result)
            return f"{result.text_nospecial}_{result.language}"
    
    def transcribe_np(self, audio: np.ndarray) -> str:
        # 同步实现，供基类 fallback 用
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, 16000)
            result = self.model(dolphin.load_audio(tmp.name))
            # print(result)
            return f"{result.text_nospecial}_{result.language}"