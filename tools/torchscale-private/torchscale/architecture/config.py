# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]


class EncoderConfig(object):
    def __init__(self, **kwargs):
        self.encoder_embed_dim = kwargs.pop("encoder_embed_dim", 768)
        self.encoder_attention_heads = kwargs.pop("encoder_attention_heads", 12)
        self.encoder_ffn_embed_dim = kwargs.pop("encoder_ffn_embed_dim", 3072)
        self.encoder_layers = kwargs.pop("encoder_layers", 12)
        self.encoder_normalize_before = kwargs.pop("encoder_normalize_before", True)
        self.normalize_output = kwargs.pop("normalize_output", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.bert_init = kwargs.pop("bert_init", False)
        self.multiway = kwargs.pop("multiway", False)
        self.share_encoder_input_output_embed = kwargs.pop(
            "share_encoder_input_output_embed", False
        )
        self.max_source_positions = kwargs.pop("max_source_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        self.layernorm_eps = kwargs.pop("layernorm_eps", 1e-5)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Vision
        self.img_size = kwargs.pop("img_size", 224)
        self.patch_size = kwargs.pop("patch_size", 16)
        self.in_chans = kwargs.pop("in_chans", 3)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.xpos_rel_pos = kwargs.pop("xpos_rel_pos", False)
        self.xpos_scale_base = kwargs.pop("xpos_scale_base", 512)
        self.flash_attention = kwargs.pop("flash_attention", False)
        self.model_parallel_size = kwargs.pop("model_parallel_size", 1)
        self.group_norm_size = kwargs.pop("group_norm_size", 1)

        if self.deepnorm:
            self.encoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.encoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)


class DecoderConfig(object):
    def __init__(self, **kwargs):
        self.decoder_embed_dim = kwargs.pop("decoder_embed_dim", 768)
        self.decoder_attention_heads = kwargs.pop("decoder_attention_heads", 12)
        self.decoder_ffn_embed_dim = kwargs.pop("decoder_ffn_embed_dim", 3072)
        self.decoder_layers = kwargs.pop("decoder_layers", 12)
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top_k = kwargs.pop("moe_top_k", 1)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", True
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.bert_init = kwargs.pop("bert_init", False)
        self.multiway = kwargs.pop("multiway", False)
        self.share_decoder_input_output_embed = kwargs.pop(
            "share_decoder_input_output_embed", False
        )
        self.max_target_positions = kwargs.pop("max_target_positions", 2048)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        self.layernorm_eps = kwargs.pop("layernorm_eps", 1e-5)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.xpos_rel_pos = kwargs.pop("xpos_rel_pos", False)
        self.xpos_scale_base = kwargs.pop("xpos_scale_base", 512)
        self.flash_attention = kwargs.pop("flash_attention", False)
        self.input_bits = kwargs.pop('input_bits', 8)
        self.input_quant_method = kwargs.pop('input_quant_method', 'elastic')
        self.weight_bits = kwargs.pop('weight_bits', 1)
        self.weight_quant_method = kwargs.pop('weight_quant_method', 'bwn')
        self.weight_featurewise = kwargs.pop('weight_featurewise', False)
        self.bmt = kwargs.pop('bmt', False)
        self.quant_ffn_only = kwargs.pop('quant_ffn_only', False)
        self.hadamard_group = kwargs.pop('hadamard_group', False)
        self.blockwise_quant = kwargs.pop('blockwise_quant', False)
        self.model_parallel_size = kwargs.pop("model_parallel_size", 1)
        self.group_norm_size = kwargs.pop("group_norm_size", 1)
        self.binary_attn = kwargs.pop("binary_attn", False)
        self.weight_blocksize = kwargs.pop("weight_blocksize", "-1,-1")
        self.grad_act = kwargs.pop("grad_act", False)
        self.weight_blockscale = kwargs.pop("weight_blockscale", 'none')
        self.no_bias = kwargs.pop("no_bias", False)
        self.rotary_embed = kwargs.pop("rotary_embed", False)
        self.rms_norm = kwargs.pop("rms_norm", False)
        self.n_kv_heads = kwargs.pop("n_kv_heads", self.decoder_attention_heads)
        self.binary_query = kwargs.pop("binary_query", False)
        self.moe_second_expert_threshold = kwargs.pop("moe_second_expert_threshold", 0.0)
        self.moe_second_expert_threshold_warmup = kwargs.pop("moe_second_expert_threshold_warmup", 0)
        self.moe_second_expert_threshold_init = kwargs.pop("moe_second_expert_threshold_init", 1e-07)
        self.binary_key = kwargs.pop("binary_key", False)
        self.key_bits = kwargs.pop("key_bits", 1)
        self.key_quant_method = kwargs.pop("key_quant_method", "bwn")
        self.moe_expert_noise_threshold = kwargs.pop("moe_expert_noise_threshold", 0.0)
        self.moe_expert_noise_threshold_warmup = kwargs.pop("moe_expert_noise_threshold_warmup", 0)
        self.moe_expert_noise_threshold_init = kwargs.pop("moe_expert_noise_threshold_init", 1e-07)
        self.moe_ffn_dim = kwargs.pop("moe_ffn_dim", -1)
        self.binary_routing = kwargs.pop("binary_routing", False)
        self.key_norm = kwargs.pop("key_norm", False)
        self.smoothquant = kwargs.pop('smoothquant', False)
        self.smoothquant_alpha = kwargs.pop('smoothquant_alpha', 0.5)
        self.ffn_bits = kwargs.pop('ffn_bits', -1)
        self.ffn_quant_method = kwargs.pop('ffn_quant_method', "")
        self.attn_bits = kwargs.pop('attn_bits', 32)
        self.attn_quant_method = kwargs.pop('attn_quant_method', "attn_absmax_per_token")
        self.absmean_alpha = kwargs.pop("absmean_alpha", 1.0)
        self.fc2_bits = kwargs.pop('fc2_bits', -1)
        self.quant_ffn_output = kwargs.pop("quant_ffn_output", False)
        self.input_absmean_alpha = kwargs.pop("input_absmean_alpha", 1.0)
        self.fc2_quant_method = kwargs.pop('fc2_quant_method', "")
        self.quant_before_rope = kwargs.pop("quant_before_rope", False)
        self.query_bits = kwargs.pop("query_bits", 32)
        self.attn_input_absmean_scale = kwargs.pop("attn_input_absmean_scale", 1.0)
        self.attn_quant_symmetric = kwargs.pop("attn_quant_symmetric", False)
        self.use_quant_for_activation = kwargs.pop("use_quant_for_activation", False)
        self.fc2_input_absmean_scale = kwargs.pop("fc2_input_absmean_scale", -1.0)
        self.negative_slope = kwargs.pop("negative_slope", -1.0)
        self.sparse_blocksize = kwargs.pop("sparse_blocksize", 16)
        self.fc2_sparse_blocksize = kwargs.pop("fc2_sparse_blocksize", -1)
        self.sparse_blocksize = kwargs.pop("sparse_blocksize", 16)
        self.fc2_sparse_blocksize = kwargs.pop("fc2_sparse_blocksize", -1)
        self.sparse_ratio = kwargs.pop("sparse_ratio", 0.4)
        self.fc2_sparse_ratio = kwargs.pop("fc2_sparse_ratio", -1.0)
        self.relu_squared = kwargs.pop("relu_squared", False)
        self.glu = kwargs.pop("glu", False)
        self.nozero_rmsnorm = kwargs.pop("nozero_rmsnorm", False)
        self.sparse_alpha = kwargs.pop("sparse_alpha", 1.0)
        self.sparse_before_quant = kwargs.pop("sparse_before_quant", False)
        self.moe_lora_rank = kwargs.pop("moe_lora_rank", -1)
        self.partial_rotary_factor = kwargs.pop("partial_rotary_factor", 1.0)
        self.kv_quant_group = kwargs.pop("kv_quant_group", 1)

        self.mhmoe_heads_number = kwargs.pop("mhmoe_heads_number", 1)

        if self.deepnorm:
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.decoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)


class EncoderDecoderConfig(object):
    def __init__(self, **kwargs):
        self.encoder_embed_dim = kwargs.pop("encoder_embed_dim", 768)
        self.encoder_attention_heads = kwargs.pop("encoder_attention_heads", 12)
        self.encoder_ffn_embed_dim = kwargs.pop("encoder_ffn_embed_dim", 3072)
        self.encoder_layers = kwargs.pop("encoder_layers", 12)
        self.encoder_normalize_before = kwargs.pop("encoder_normalize_before", True)
        self.decoder_embed_dim = kwargs.pop("decoder_embed_dim", 768)
        self.decoder_attention_heads = kwargs.pop("decoder_attention_heads", 12)
        self.decoder_ffn_embed_dim = kwargs.pop("decoder_ffn_embed_dim", 3072)
        self.decoder_layers = kwargs.pop("decoder_layers", 12)
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.bert_init = kwargs.pop("bert_init", False)
        self.multiway = kwargs.pop("multiway", False)
        self.share_all_embeddings = kwargs.pop("share_all_embeddings", False)
        self.share_decoder_input_output_embed = kwargs.pop(
            "share_decoder_input_output_embed", False
        )
        self.max_source_positions = kwargs.pop("max_source_positions", 1024)
        self.max_target_positions = kwargs.pop("max_target_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        self.layernorm_eps = kwargs.pop("layernorm_eps", 1e-5)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.xpos_rel_pos = kwargs.pop("xpos_rel_pos", False)
        self.xpos_scale_base = kwargs.pop("xpos_scale_base", 512)
        self.flash_attention = kwargs.pop("flash_attention", False)
        self.model_parallel_size = kwargs.pop("model_parallel_size", 1)
        self.group_norm_size = kwargs.pop("group_norm_size", 1)

        if self.deepnorm:
            self.encoder_normalize_before = False
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.encoder_normalize_before = True
            self.decoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)
