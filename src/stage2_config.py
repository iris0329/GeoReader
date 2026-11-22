"""Stage2 training, saving, and evaluation share common constant patterns."""

CAMERA_INTERPRETER_TRAINING_MODE_FROZEN = "frozen_checkpoint"
CAMERA_INTERPRETER_TRAINING_MODE_DIRECT = "direct_random"
# Stage 1 weights are only responsible for initialization; Stage 2 QA loss continues to update the Interpreter.
CAMERA_INTERPRETER_TRAINING_MODE_STAGE1_TRAINABLE = "stage1_checkpoint_trainable"
CAMERA_INTERPRETER_TRAINING_MODE_CHOICES = (
    CAMERA_INTERPRETER_TRAINING_MODE_FROZEN,
    CAMERA_INTERPRETER_TRAINING_MODE_DIRECT,
    CAMERA_INTERPRETER_TRAINING_MODE_STAGE1_TRAINABLE,
)
DEPTH_INTERPRETER_TRAINING_MODE_FROZEN = "frozen_checkpoint"
DEPTH_INTERPRETER_TRAINING_MODE_DIRECT = "direct_random"
# Two paths using the same mode name to avoid Camera/Depth checkpoint contracts relying on context to infer semantics.
DEPTH_INTERPRETER_TRAINING_MODE_STAGE1_TRAINABLE = "stage1_checkpoint_trainable"
DEPTH_INTERPRETER_TRAINING_MODE_CHOICES = (
    DEPTH_INTERPRETER_TRAINING_MODE_FROZEN,
    DEPTH_INTERPRETER_TRAINING_MODE_DIRECT,
    DEPTH_INTERPRETER_TRAINING_MODE_STAGE1_TRAINABLE,
)
OUTPUT_GATE_TYPE_SCALAR = "scalar"
OUTPUT_GATE_TYPE_NONE = "none"
OUTPUT_GATE_TYPE_DYNAMIC = "dynamic"
OUTPUT_GATE_TYPE_SOFTMAX = "softmax"
OUTPUT_GATE_TYPE_SOFTMAX_THREEWAY = "softmax_threeway"
OUTPUT_GATE_TYPE_INDEPENDENT_SIGMOID = "independent_sigmoid"
# Question-conditioned High-dimensional Gate: Each modality outputs hidden_size channels of weights, rather than a scalar.
OUTPUT_GATE_TYPE_QUESTION_CHANNELWISE_SIGMOID = "question_channelwise_sigmoid"
# FiLM applies question-conditioned channel scaling and shifting through a residual adapter.
OUTPUT_GATE_TYPE_QUESTION_FILM = "question_film"
OUTPUT_GATE_TYPE_FIXED_SCALE = "fixed_scale"
OUTPUT_GATE_TYPE_SOFTMAX_FAMILY = (
    OUTPUT_GATE_TYPE_SOFTMAX,
    OUTPUT_GATE_TYPE_SOFTMAX_THREEWAY,
)
# These three Gate types share the raw delta, Adapter, and H18/H36 injection into the main flow; checkpoint semantics remain independently separate.
OUTPUT_GATE_TYPE_SPATIAL_ROUTER_FAMILY = (
    *OUTPUT_GATE_TYPE_SOFTMAX_FAMILY,
    OUTPUT_GATE_TYPE_INDEPENDENT_SIGMOID,
    OUTPUT_GATE_TYPE_QUESTION_CHANNELWISE_SIGMOID,
    OUTPUT_GATE_TYPE_QUESTION_FILM,
)
# fixed_scale reuses the Spatial Adapter and learned empty slots without Router/Gate parameters.
OUTPUT_GATE_TYPE_SPATIAL_ADAPTER_FAMILY = (
    *OUTPUT_GATE_TYPE_SPATIAL_ROUTER_FAMILY,
    OUTPUT_GATE_TYPE_FIXED_SCALE,
)
OUTPUT_GATE_TYPE_CHOICES = (
    OUTPUT_GATE_TYPE_NONE,
    OUTPUT_GATE_TYPE_SCALAR,
    OUTPUT_GATE_TYPE_DYNAMIC,
    OUTPUT_GATE_TYPE_SOFTMAX,
    OUTPUT_GATE_TYPE_SOFTMAX_THREEWAY,
    OUTPUT_GATE_TYPE_INDEPENDENT_SIGMOID,
    OUTPUT_GATE_TYPE_QUESTION_CHANNELWISE_SIGMOID,
    OUTPUT_GATE_TYPE_QUESTION_FILM,
    OUTPUT_GATE_TYPE_FIXED_SCALE,
)
SPATIAL_FUSION_TYPE_LATENT_TOKENS = "latent_tokens"
SPATIAL_FUSION_TYPE_IMAGE_RESIDUAL = "image_residual"
# frozen Stage1 Depth patch memory writes back RGB patch to the same position after zero-init Linear; does not generate Depth token block.
SPATIAL_FUSION_TYPE_DEPTH_PATCH_RESIDUAL = "depth_patch_residual"
# The Stage1 output of Direct Patch Depth already contains complete Qwen embeddings per patch; only the position of the second Depth block can be altered between the two types.
SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_BEFORE_QUESTION = (
    "direct_patch_depth_twopass_before_question"
)
SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_AFTER_QUESTION = (
    "direct_patch_depth_twopass_after_question"
)
# direct_embedding_tokens: The Interpreter output is already a complete Qwen embedding and must directly overwrite the pad slot; no additional base/Gate/Adapter can be added.
SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TOKENS = "direct_embedding_tokens"
# direct_embedding_twopass: First pass only uses RGB+Question, second pass uses H36 Softmax scaling to fully expand the Direct embedding.
SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TWOPASS = "direct_embedding_twopass"
# Frame Reader: First pass reads RGB+Question H36; second pass scales or interpolates full Direct embedding frame by frame.
SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TWOPASS_PERFRAME_READER = (
    "direct_embedding_twopass_perframe_reader"
)
SPATIAL_FUSION_TYPE_CHOICES = (
    SPATIAL_FUSION_TYPE_LATENT_TOKENS,
    SPATIAL_FUSION_TYPE_IMAGE_RESIDUAL,
    SPATIAL_FUSION_TYPE_DEPTH_PATCH_RESIDUAL,
    SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_BEFORE_QUESTION,
    SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_AFTER_QUESTION,
    SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TOKENS,
    SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TWOPASS,
    SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TWOPASS_PERFRAME_READER,
)
SOFTMAX_SPATIAL_INJECTION_LAYER = 18
SPATIAL_GATE_SCHEDULE_ONE_PASS = "one_pass_h18_mid18"
SPATIAL_GATE_SCHEDULE_TWO_PASS = "two_pass_h36_input"
SPATIAL_GATE_SCHEDULE_ONE_PASS_FIXED_INPUT = "one_pass_fixed_input"
SPATIAL_GATE_SCHEDULE_CHOICES = (
    SPATIAL_GATE_SCHEDULE_ONE_PASS,
    SPATIAL_GATE_SCHEDULE_TWO_PASS,
    SPATIAL_GATE_SCHEDULE_ONE_PASS_FIXED_INPUT,
)
SPATIAL_SLOT_FUSION_RESIDUAL_ADD = "residual_add"
SPATIAL_SLOT_FUSION_REPLACEMENT = "replacement"
# Direct, two passes, no padding base, not base/candidate interpolation; final payload is only gate × H_direct.
SPATIAL_SLOT_FUSION_DIRECT_GATE_SCALE = "direct_gate_scale"
SPATIAL_SLOT_FUSION_CHOICES = (
    SPATIAL_SLOT_FUSION_RESIDUAL_ADD,
    SPATIAL_SLOT_FUSION_REPLACEMENT,
    SPATIAL_SLOT_FUSION_DIRECT_GATE_SCALE,
)
SOFTMAX_QUESTION_MASK_SOURCE = "tokenizer_offset_mapping_original_question"
MISSING_POSE_TOKEN_POLICY_DROP = "drop"
MISSING_POSE_TOKEN_POLICY_KEEP_ZERO_DELTA = "keep_zero_delta"
MISSING_POSE_TOKEN_POLICY_CHOICES = (
    MISSING_POSE_TOKEN_POLICY_DROP,
    MISSING_POSE_TOKEN_POLICY_KEEP_ZERO_DELTA,
)
SPATIAL_TOKEN_POSITION_CHOICES = ("before_image", "before_question", "after_question")
