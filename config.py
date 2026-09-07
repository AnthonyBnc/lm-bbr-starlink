import os


class Config:
    _base_dir = '' if 'llm_framework' in os.getcwd() else 'llm_framework/'
    _home_dir = os.path.expanduser('~')

    data_dir = _base_dir + 'data/'
    results_dir = data_dir + 'results/'
    exp_pools_dir = data_dir + 'exp_pools/'

    # plm special
    plm_types = ['gpt2', 'llama2', 't5', 'llama3', 'smollm2', 'gpt_neo', 'gemma3', 'qwen3']
    plm_sizes = ['xxs', 'xs', 'small', 'base', 'large', 'xl', 'xxl']  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.

    # plm_dir = _base_dir + path to your downloaded_plms
    plm_dir = os.environ.get('LM_BBR_PLM_DIR', os.path.join(_home_dir, 'models', 'downloaded_plms'))
    local_model_root = os.environ.get(
        'LM_BBR_LOCAL_MODEL_ROOT',
        os.path.join(_home_dir, 'models', 'lm-bbr-starlink'),
    )
    modern_model_registry = {
        'gemma_3_270m': {
            'plm_type': 'gemma3',
            'plm_size': None,
            'hf_id': 'google/gemma-3-270m',
            'local_dir': 'gemma-3-270m',
            'revision': '9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1',
            'release_date': '2025-08-14',
            'loader_status': 'registered_pending_smoke',
            'preferred_dtype': 'bfloat16',
        },
        'granite_4_0_350m': {
            'plm_type': None,
            'plm_size': None,
            'hf_id': 'ibm-granite/granite-4.0-350m',
            'local_dir': 'granite-4.0-350m',
            'revision': 'bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2',
            'release_date': '2025-10-28',
            'loader_status': 'registered_pending_smoke',
            'preferred_dtype': 'bfloat16',
        },
        'pleias_rag_350m': {
            'plm_type': None,
            'plm_size': None,
            'hf_id': 'PleIAs/Pleias-RAG-350M',
            'local_dir': 'Pleias-RAG-350M',
            'revision': 'db001b29a33532583d14979c3c38ef04b6d59352',
            'release_date': '2025-04-07',
            'loader_status': 'registered_pending_smoke',
            'preferred_dtype': 'bfloat16',
        },
        'lfm2_5_350m': {
            'plm_type': None,
            'plm_size': None,
            'hf_id': 'LiquidAI/LFM2.5-350M',
            'local_dir': 'LFM2.5-350M',
            'revision': '9e6c6ccf47cd318696e137d381a7ded8fe4df09f',
            'release_date': '2026-03-31',
            'loader_status': 'registered_pending_smoke',
            'preferred_dtype': 'float16',
        },
        'granite_4_0_h_350m': {
            'plm_type': None,
            'plm_size': None,
            'hf_id': 'ibm-granite/granite-4.0-h-350m',
            'local_dir': 'granite-4.0-h-350m',
            'revision': '3b17b717b8f2f5d305b0a92c1491e239aeda19c8',
            'release_date': '2025-10-28',
            'loader_status': 'registered_pending_smoke',
            'preferred_dtype': 'bfloat16',
        },
        'qwen3_5_4b_base': {
            'plm_type': 'qwen3',
            'plm_size': 'base',
            'hf_id': 'Qwen/Qwen3.5-4B-Base',
            'local_dir': 'Qwen3.5-4B-Base',
            'revision': '1001bb4d826a52d1f399e183466143f4da7b741b',
            'release_date': '2026-02-15',
            'loader_status': 'generic_auto_model_smoke_passed',
            'preferred_dtype': 'bfloat16',
        },
        'gemma_3_4b_pt': {
            'plm_type': 'gemma3',
            'plm_size': 'base',
            'hf_id': 'google/gemma-3-4b-pt',
            'local_dir': 'gemma-3-4b-pt',
            'revision': 'cc012e0a6d0787b4adcc0fa2c4da74402494554d',
            'release_date': '2025-03-10',
            'loader_status': 'generic_auto_model_smoke_passed',
            'preferred_dtype': 'bfloat16',
        },
        'llama_3_2_3b': {
            'plm_type': 'llama3',
            'plm_size': 'base',
            'hf_id': 'meta-llama/Llama-3.2-3B',
            'local_dir': 'Llama-3.2-3B',
            'revision': '13afe5124825b4f3751f836b40dafda64c1ed062',
            'release_date': '2024-09-25',
            'loader_status': 'generic_auto_model_smoke_passed',
            'preferred_dtype': 'float16',
        },
        'lfm2_5_2_6b': {
            'plm_type': None,
            'plm_size': None,
            'hf_id': 'LiquidAI/LFM2.5-2.6B',
            'local_dir': 'LFM2.5-2.6B',
            'revision': 'a4e00e83c0979ee9deb88d04b6360599fa956656',
            'release_date': '2025-11-28',
            'loader_status': 'generic_auto_model_smoke_passed',
            'preferred_dtype': 'float16',
        },
        'olmo_3_1025_7b': {
            'plm_type': None,
            'plm_size': None,
            'hf_id': 'allenai/Olmo-3-1025-7B',
            'local_dir': 'Olmo-3-1025-7B',
            'revision': 'a81bae42db3975be1671e27b9c9a56da1a9f980f',
            'release_date': '2025-11-20',
            'loader_status': 'generic_auto_model_smoke_passed',
            'preferred_dtype': 'bfloat16',
        },
    }
    lora_defaults = {
        'construction_rank': 8,
        'alpha': 32,
        'dropout': 0.05,
        'bias': 'none',
        'task_type': 'FEATURE_EXTRACTION',
        'status': 'exploratory_pending_training_policy_approval',
    }
    quantum_defaults = {
        'parent_model_key': 'lfm2_5_350m',
        'n_qubits': 8,
        'depth': 2,
        'encoding': 'bounded_ry_angle_encoding_reuploaded_each_layer',
        'ansatz': 'data_reuploading_ry',
        'entanglement': 'cnot_ring',
        'measurement': 'per_qubit_pauli_z_expectation',
        'framework': 'qiskit',
        'estimator': 'qiskit.primitives.StatevectorEstimator',
        'simulator': 'qiskit_statevector_estimator',
        'default_precision': 0.0,
        'shots': None,
        'status': 'frozen_for_under400m_follow_on_adr_0017',
    }
    modern_lora_registry = {
        'gemma_3_270m': {
            'target_modules': ['q_proj', 'v_proj'],
            'expected_adapter_modules': 36,
            'notes': 'Attention Q/V projections across all 18 Gemma 3 text-transformer layers.',
        },
        'granite_4_0_350m': {
            'target_modules': ['q_proj', 'v_proj'],
            'expected_adapter_modules': 56,
            'notes': 'Attention Q/V projections across all 28 Granite transformer layers.',
        },
        'pleias_rag_350m': {
            'target_modules': ['q_proj', 'v_proj'],
            'expected_adapter_modules': 52,
            'notes': 'Attention Q/V projections across all 26 Llama-architecture layers.',
        },
        'lfm2_5_350m': {
            'target_modules': ['q_proj', 'v_proj', 'in_proj', 'out_proj'],
            'expected_adapter_modules': 38,
            'notes': 'Attention Q/V/output plus hybrid convolution input/output projections.',
        },
        'granite_4_0_h_350m': {
            'target_modules': ['q_proj', 'v_proj', 'in_proj', 'out_proj'],
            'expected_adapter_modules': 64,
            'notes': 'Attention Q/V plus hybrid Mamba input/output projections.',
        },
        'qwen3_5_4b_base': {
            'target_modules': r'.*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|v_proj)|linear_attn\.(?:in_proj_qkv|out_proj))',
            'expected_adapter_modules': 64,
            'notes': 'Q/V for 8 full-attention layers; fused QKV input and output for 24 linear-attention layers.',
        },
        'gemma_3_4b_pt': {
            'target_modules': r'.*language_model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj)',
            'expected_adapter_modules': 68,
            'notes': 'Text Q/V only; vision tower is excluded.',
        },
        'llama_3_2_3b': {
            'target_modules': ['q_proj', 'v_proj'],
            'expected_adapter_modules': 56,
            'notes': 'Attention Q/V, matching the legacy Llama LoRA scope.',
        },
        'lfm2_5_2_6b': {
            'target_modules': ['q_proj', 'v_proj', 'in_proj', 'out_proj'],
            'expected_adapter_modules': 68,
            'notes': 'Attention Q/V plus hybrid convolution input/output projections.',
        },
        'olmo_3_1025_7b': {
            'target_modules': ['q_proj', 'v_proj'],
            'expected_adapter_modules': 64,
            'notes': 'Attention Q/V.',
        },
    }
    plm_ft_dir = _base_dir + 'data/ft_plms'
    plm_embed_sizes = {
        'gpt2': {
            'base': 1024,
            'small': 768,
            'large': 1280,
            'xl': 1600,
        },
        'llama2': {
            'base': 4096,
        },
        't5': {
            'base': 768,
            'small': 512,
            'large': 4096,
            'xl': 2048,
        },
        'llava': {
            'base': 4096,
        },
        'mistral': {
            'base': 4096,
        },
        'opt': {
            'large': 5120,
            'base': 4096,
            'small': 2560,
            'xs': 2048,
            'xxs': 512,
        },
        'llama3': {
            'base': 3072,
        },
        'llama4': {
            'base': 4096,
        },
        'deepseek': {
            'base': 4096,
        },
        'gemma3': {
            'base': 640,
        },
        'qwen3': {
            'base': 1024,
        },
        'smollm2': {
            'base': 960,
        },
        'pythia': {
            'base': 1024,
        },
        'gpt_neo': {
            'base': 768,
        },
    }
    plm_layer_sizes = {
        'gpt2': {
            'base': 24,
            'small': 12,
            'large': 36,
            'xl': 48
        },
        'llama2': {
            'base': 32,
        },
        't5': { 
            'base': 12,
            'small': 6,
            'large': 24,
            'xl': 24
        },
        'llava': {
            'base': 32,
        },
        'mistral': {
            'base': 32,
        },
        'opt': {
            'large': 40,
            'base': 32,
            'small': 32,
            'xs': 32,
            'xxs': 16,
        },
        'llama3': {
            'base': 28,
        },
        'llama4': {
            'base': 48,
        },
        'deepseek': {
            'base': 32,
        },
        'gemma3': {
            'base': 18,
        },
        'qwen3': {
            'base': 28,
        },
        'smollm2': {
            'base': 32,
        },
        'pythia': {
            'base': 24,
        },
        'gpt_neo': {
            'base': 12,
        },
    }

    @classmethod
    def get_registered_model(cls, model_key):
        return cls.modern_model_registry[model_key]

    @classmethod
    def get_registered_model_path(cls, model_key):
        model_info = cls.get_registered_model(model_key)
        return os.path.join(cls.local_model_root, model_info['local_dir'])


cfg = Config()
