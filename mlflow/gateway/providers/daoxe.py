from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class DaoxeProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "DaoXE"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.daoxe.com/v1"
