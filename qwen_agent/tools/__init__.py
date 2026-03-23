# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from typing import Iterable

from .base import TOOL_REGISTRY, BaseTool
from .mcp_manager import MCPManager

__all__ = [
    'BaseTool',
    'TOOL_REGISTRY',
    'MCPManager',
]


def _safe_import(module_name: str, exported_names: Iterable[str]) -> None:
    try:
        module = importlib.import_module(f'.{module_name}', __name__)
    except ImportError:
        return

    for name in exported_names:
        globals()[name] = getattr(module, name)
        __all__.append(name)


_safe_import('amap_weather', ['AmapWeather'])
_safe_import('code_interpreter', ['CodeInterpreter'])
_safe_import('doc_parser', ['DocParser'])
_safe_import('extract_doc_vocabulary', ['ExtractDocVocabulary'])
_safe_import('image_gen', ['ImageGen'])
_safe_import('python_executor', ['PythonExecutor'])
_safe_import('retrieval', ['Retrieval'])
_safe_import('image_zoom_in_qwen3vl', ['ImageZoomInToolQwen3VL'])
_safe_import('image_search', ['ImageSearch'])
_safe_import('search_tools', ['FrontPageSearch', 'HybridSearch', 'KeywordSearch', 'VectorSearch'])
_safe_import('simple_doc_parser', ['SimpleDocParser'])
_safe_import('storage', ['Storage'])
_safe_import('web_extractor', ['WebExtractor'])
_safe_import('web_search', ['WebSearch'])
