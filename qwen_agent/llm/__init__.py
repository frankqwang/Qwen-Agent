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

import copy
import importlib
from typing import Union

from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError

__all__ = [
    'BaseChatModel',
    'get_chat_model',
    'ModelServiceError',
]


def _safe_import(module_name: str, exported_names: list[str]) -> None:
    try:
        module = importlib.import_module(f'.{module_name}', __name__)
    except ImportError:
        return

    for name in exported_names:
        globals()[name] = getattr(module, name)
        __all__.append(name)


_safe_import('azure', ['TextChatAtAzure'])
_safe_import('oai', ['TextChatAtOAI'])
_safe_import('openvino', ['OpenVINO'])
_safe_import('qwen_dashscope', ['QwenChatAtDS'])
_safe_import('qwenaudio_dashscope', ['QwenAudioChatAtDS'])
_safe_import('qwenomni_oai', ['QwenOmniChatAtOAI'])
_safe_import('qwenvl_dashscope', ['QwenVLChatAtDS'])
_safe_import('qwenvl_oai', ['QwenVLChatAtOAI'])
_safe_import('qwenvlo_dashscope', ['QwenVLoChatAtDS'])
_safe_import('transformers_llm', ['Transformers'])


def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    """The interface of instantiating LLM objects.

    Args:
        cfg: The LLM configuration, one example is:
          cfg = {
              # Use the model service provided by DashScope:
              'model': 'qwen-max',
              'model_server': 'dashscope',

              # Use your own model service compatible with OpenAI API:
              # 'model': 'Qwen',
              # 'model_server': 'http://127.0.0.1:7905/v1',

              # (Optional) LLM hyper-parameters:
              'generate_cfg': {
                  'top_p': 0.8,
                  'max_input_tokens': 6500,
                  'max_retries': 10,
              }
          }

    Returns:
        LLM object.
    """
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            if model_type in ('oai', 'qwenvl_oai'):
                if cfg.get('model_server', '').strip() == 'dashscope':
                    cfg = copy.deepcopy(cfg)
                    cfg['model_server'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            return LLM_REGISTRY[model_type](cfg)
        else:
            raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    # Deduce model_type from model and model_server if model_type is not provided:

    if 'azure_endpoint' in cfg:
        model_type = 'azure'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if 'model_server' in cfg:
        if cfg['model_server'].strip().startswith('http'):
            model_type = 'oai'
            cfg['model_type'] = model_type
            return LLM_REGISTRY[model_type](cfg)

    model = cfg.get('model', '')

    if '-vl' in model.lower():
        model_type = 'qwenvl_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if '-audio' in model.lower():
        model_type = 'qwenaudio_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if 'qwen' in model.lower():
        model_type = 'qwen_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    raise ValueError(f'Invalid model cfg: {cfg}')

