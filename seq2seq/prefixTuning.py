# from transformers import Trainer
import torch
from torch.nn import Embedding, Sequential, Parameter, Module

from transformers import (
    GPT2PreTrainedModel,
    PreTrainedTokenizer,
    GPT2Tokenizer,
    PretrainedBartModel,
)
from torch import nn, Tensor
from typing import Optional, Literal, List, Tuple


class PrefixTuning(PretrainedBartModel):
    """Classification Head for  transformer encoders

    Notes:
        完整的 prefix prompt 训练分为两阶段：
          - 第 1 阶段（低数据初始化训练）：预对齐预训练模型特征（初始化训练），目的是让前缀 prompt 生成的特征，
                对齐预训练模型（GPT-2）的原生注意力缓存特征，避免前缀随机初始化带来的训练不稳定性，提升后续下游任务的收敛速度
                方法：lowdata_init_not_need_tokenize_train、lowdata_init_need_tokenize_train
          - 第 2 阶段：适配下游任务（正式微调）
          - 第 1 阶段只有 self.control_trans 会训练；第 2 阶段 self.wte、self.control_trans 会训练


    Attributes:
        preseqlen (int): Prompt 的序列长度（生成的 Prompt 包含多少个「虚拟 token」）（前缀优化序列长度？）
        optim_prefix (bool): 是否前缀优化
        use_infix (bool): 是否使用 infix
        use_deep (bool):  是否使用深度模式
        n_layer (int):
        match_n_layer (int): 匹配的 GPT-2 模型层数（如 24 层，需与目标 GPT-2 一致）。（解码器层数？）
        match_n_head (int):  匹配的 GPT-2 注意力头数（如 16 头，需与目标 GPT-2 一致）（解码器注意力头数？）
        match_n_embd (int):  每个注意力头的维度（如 64，match_n_embd = hidden_size / match_n_head）。
        n_embd (int):  嵌入的维度
        task_mode (str):  任务模式
        tuning_mode (str):  微调模式：prefixtune
        train_weights (bool):  是否训练权重
        format_mode (str):  前缀与输入的拼接格式 ["cat", "infix", "peek", "nopeek"]
        prefix_dropout (float): 前缀 dropout 值
        dropout(Module): dropout 层（防止过拟合，对生成的 Prompt 特征进行随机失活）
        init_random (bool): 是否随机初始化
        mid_dim (int): 隐藏层维度
        lowdata (bool): 是否低数据场景？
        lowdata_token (str): 低数据场景的初始化 token
        task_mode (str): 任务模式,，必须为以下值：
                - writingPrompts
                - webnlg
                - triples
                - data2text
                - dataless
        mode_para (int): 运行模式开关参数 [0, 1, 2(writingPrompts、webnlg、triples、data2text),3, 4]
            - 0: for data2text Instruction based, just optimize a set of parameters （self.mode_para=0 and optim_prefix == True for Instruction based.）
            - 1: for dataless
            - 2：data2text having a variable length input prefix parametrization. or for (4) topic
            - 3: OLD VERSION: many parameters.
            - 4：topic/keyword/attributes..
        wte (Embedding): 自注意力场景的词嵌入层（Word Token Embedding）(权重嵌入?)
        wte2 (Embedding): 交叉注意力场景的词嵌入层（Word Token Embedding）(权重嵌入?)
        wte_enc (Embedding): 编码器的 Prompt 嵌入层（encoder prefix 的权重嵌入）
        control_trans (Sequential): 特征转换模块：生成自注意力的 prompt 特征（是一个 MLP）
        control_trans2 (Sequential): 特征转换模块：生成交叉注意力的 prompt 特征（是一个 MLP）
        control_trans_enc (Sequential): 特征转换模块：生成编码器的 prompt 特征（是一个 MLP）
        use_cross_prefix (bool): 是否启用交叉注意力 Prompt
        use_encoder_prefix (bool): 是否启用编码器 Prompt
        embMatch (bool): 是否直接使用 control_code 作为嵌入，True 则跳过词嵌入步骤
        input_embs (Tensor): 预定义的 Prompt 嵌入向量（形状 [1, preseqlen, emb_size] 或 [preseqlen, emb_size]），
            无需通过词嵌入层转换，直接作为输入。
            是提前定义好的嵌入向量（而非离散 token 索引），可能是随机初始化后可训练的参数，
              或通过其他方式预优化的向量（如从预训练模型中提取的特征）。

    """

    preseqlen: int = 5
    optim_prefix: bool = False
    use_infix: bool = False
    use_deep: bool = False
    n_layer: int
    match_n_layer: int
    match_n_head: int
    match_n_embd: int
    n_embd: int
    task_mode: Literal["writingPrompts", "webnlg", "triples", "data2text", "dataless"]
    tuning_mode: str
    train_weights: bool
    format_mode: Literal["cat", "infix", "peek", "nopeek"] = "cat"
    prefix_dropout: float = 0.0
    dropout: Module
    init_random: bool = False
    mid_dim: int = 512
    lowdata: bool = False
    lowdata_token: Optional[str] = None
    mode_para: Literal[0, 1, 2, 3, 4] = 1
    wte: Embedding
    wte2: Embedding
    wte_enc: Embedding
    control_trans: Sequential
    control_trans2: Sequential
    control_trans_enc: Sequential
    use_cross_prefix: bool
    use_encoder_prefix: bool = True
    input_embs: Tensor
    embMatch: bool

    def __init__(
        self,
        config,
        model_gpt2,
        optim_prefix: bool = False,
        preseqlen: int = 5,
        use_infix: bool = False,
        deep_param=False,
    ):
        super().__init__(config)
        print("under the PrefixTuning model")

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head
        # 1. 是否开启前缀优化（核心开关）
        if hasattr(config, "optim_prefix"):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix
        # 2. 前缀长度（preseqlen：prefix sequence length，核心超参数）
        if hasattr(config, "preseqlen") and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        # 3. 是否使用中缀（infix：前缀是前置，中缀是插入到序列中间，小众场景）
        if hasattr(config, "use_infix"):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix
        # 4. 是否使用深层参数化（deep_param：投影层是否增加隐藏层）
        if hasattr(config, "use_deep"):
            self.use_deep = config.use_deep == "yes"
        else:
            self.use_deep = False

        deep_param = self.use_deep
        # 5. 调优模式（默认 prefixtune，确保仅前缀优化）
        if hasattr(config, "_my_arg_tune_mode"):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = "prefixtune"
        # 6. 任务模式（必须指定，否则报错，如 dataless、data2text 等）
        if hasattr(config, "_my_arg_task_mode"):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = "underspecified"
            assert False, "the task is underspecified"
        # 7. 其他辅助参数（训练权重、格式模式、前缀 dropout、低数据场景等）
        if hasattr(config, "train_weights"):
            self.train_weights = config.train_weights == "yes"
        else:
            assert False, "unspecified train weights"

        if hasattr(config, "format_mode"):
            self.format_mode = config.format_mode
        else:
            self.format_mode = "cat"

        # prefix_dropout: dropout rate for the prefix tuning model.
        if hasattr(config, "prefix_dropout"):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        if hasattr(config, "init_random"):
            self.init_random = config.init_random == "yes"
        else:
            self.init_random = False

        if hasattr(config, "mid_dim"):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, "lowdata"):
            self.lowdata = config.lowdata
        else:
            self.lowdata = False

        if hasattr(config, "lowdata_token"):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None

        # 根据 task_mode 映射为 mode_para 整数，对应不同的前缀生成方法（get_prompt 系列），以适配不同任务的前缀需求
        if self.task_mode == "dataless":
            self.mode_para = 1 # 无数据场景（仅依赖指令，无训练数据）
        elif (
                self.task_mode == "data2text"
                or self.task_mode == "triples"
                or self.task_mode == "webnlg"
                or self.task_mode == "writingPrompts"
        ):
            # with src and input based encoding.
            self.mode_para = 2 # 数据到文本生成（如 WebNLG 三元组转文本）
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4 # 主题/关键词/属性驱动生成（其他生成任务）
        # 分支一：非前缀优化：仅用于构建基线模型（对比前缀优化的效果），分为「微调词嵌入」和「随机初始化词嵌入基线」两种
        if not self.optim_prefix:
            # 一. 首先初始化主干模型的嵌入层权重
            # 子场景 1：微调主干模型的词嵌入层（wte）
            if self.train_weights:
                self.wte = model_gpt2.transformer.wte # 加载 GPT-2 预训练词嵌入
                for p in self.wte.parameters():
                    p.requires_grad = True # 开启词嵌入层训练（仅这部分，主体仍冻结）
            # 子场景 2：基线模型（不加载预训练词嵌入，随机初始化）
            else:
                if not self.init_random:
                    self.wte = None
                else:
                    print(
                        "the is just for baseline checking!!! We reinitialize the LM embeddings and try cat "
                        "and peek."
                    )
                    print("BASELINE" * 100)
                    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
                    print(self.wte)
            # 二. 初始化 control_trans 和 get_prompt
            # control_trans 是「低维前缀投影层」，通过「线性层 + Tanh 激活」将低维前缀嵌入映射到 GPT-2 所需的高维维度
            #   n_layer * 2 * n_embd，2 对应 key 和 value 两个注意力张量，是 Prefix Tuning 减少参数量的核心技巧
            # 不同 mode_para 初始化对应的投影层（control_trans）和前缀生成方法（get_prompt）
            # dataless 无数据场景
            if self.mode_para == 1:
                print("mode_para=1, for dataless.")
                # 低维投影层（核心：减少参数量）
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p4_infix
                else:
                    self.get_prompt = self.get_prompt_p4
            # mode_para=2 or 4, for data2text/topic generation.
            elif self.mode_para == 2 or self.mode_para == 4:
                print(
                    "mode_para=2 or 4, for (2)data2text having a variable length input prefix parametrization. or for (4) topic/keyword/attributes..."
                )
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p3_infix
                else:
                    self.get_prompt = self.get_prompt_p3
            # mode_para=3, OLD VERSION: many parameters.
            elif self.mode_para == 3:
                print("mode_para=3, OLD VERSION: many parameters.")
                self.control_trans = nn.Sequential(
                    nn.Linear(
                        config.n_embd,
                        self.preseqlen * config.n_layer * 2 * config.n_embd,
                        ),
                    nn.Tanh(),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p1_infix
                else:
                    self.get_prompt = self.get_prompt_p1
        # 分支二：前缀优化
        else:
            self.mode_para = 0 # mode_para = 0 指令式任务专用，分为「低数据场景」「普通浅层参数化」「普通深层参数化」三个场景
            print(
                "mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) "
            )
            print(
                "preseqlen is {}, under the mode of optimizing prefix directly".format(
                    self.preseqlen
                )
            )
            # 场景1. 低数据场景
            # 针对训练数据极少的场景，通过样本输入初始化前缀参数，提升模型收敛速度，分 3 种初始化方式（重点是 low_data_init=3）：
            if self.lowdata and self.lowdata_token is not None:
                low_data_init = 3
                # 三种初始化方式
                # 方式 1：利用样本输入，【使用基础模型】初始化前缀张量（直接赋值 control_trans）
                if low_data_init == 1:
                    print(
                        "IN THE LOW DATA SETTING, EXPLORE INITIALIZATION FOR DIRECT OPTIM..."
                    )
                    # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p22

                    # 参数 "gpt2-medium" 指定加载的是「中等尺寸 GPT-2 模型」的配套分词器 —— 但分词器的类类型与模型尺寸无关：
                    # 无论加载 gpt2（基础版）、gpt2-medium（中等版）、gpt2-large（大型版）还是 gpt2-xl（超大型版），
                    # 返回的都是 GPT2Tokenizer 类实例，差异仅在于分词器的词汇表（vocab）和配置（但 GPT-2 全系列共享同一套词汇表，因此实际差异极小）。
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
                    sample_text = "name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 ."
                    # src：原始数据表；tgt：目标文本
                    src, tgt = sample_text.split("||")
                    sample_input = (
                            " {} {} ".format(src, tokenizer.bos_token)
                            + tgt
                            + " {}".format(tokenizer.eos_token)
                    )
                    self.control_trans = self.lowdata_init_train1(
                        gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input
                    )
                    print(self.control_trans.shape)
                # 方式 2：初始化前缀嵌入层 + 投影层，利用样本【从零】训练初始化
                # input_tokens 是前缀的虚拟令牌索引（无需对应真实词汇表），
                # self.wte 是前缀专属嵌入层（形状 [preseqlen, n_embd]），这是 Prefix Tuning 「连续前缀嵌入」的核心载体。
                elif low_data_init == 2:
                    print(
                        "IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, need to train first"
                    )
                    self.input_tokens = torch.arange(self.preseqlen).long() # 前缀令牌索引（0~preseqlen-1）
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd) # 前缀嵌入层（可训练，核心）
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
                    )
                    self.get_prompt = self.get_prompt_p5

                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
                    # sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    sample_text = "name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 ."
                    src, tgt = sample_text.split("||")
                    sample_input = (
                            " {} {} ".format(src, tokenizer.bos_token)
                            + tgt
                            + " {}".format(tokenizer.eos_token)
                    )

                # 方式 3：预定义 token 初始化，统一与微调逻辑（最常用）
                # 初始化前缀嵌入层 + 投影层，使用 lowdata_token 而非样本初始化
                elif low_data_init == 3:
                    # use a single prepended token.
                    assert self.lowdata_token is not None
                    self.preseqlen = len(self.lowdata_token[0])
                    print(
                        "IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, low_data_init=3, "
                        "preseqlen = {} Unifying with FINETUNE".format(self.preseqlen)
                    )
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd) # 前缀嵌入层（可训练）
                    # 浅层投影层
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
                    )
                    self.get_prompt = self.get_prompt_p5

            # DIFFERENT PARAMETRIZATION:
            # 普通浅层参数化（主流）
            # 非低数据场景的默认选择，投影层为「两层线性 + 一层 Tanh」（浅层），同时支持「编码器前缀」和「交叉注意力前缀」
            #   （适配 Encoder-Decoder 结构，此处兼容 GPT-2 Decoder-only 结构）：
            elif not deep_param:
                low_data_init = 0 # 非低数据场景
                print("UNDER PARAMETRIZATION 1")
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd) # 解码器前缀嵌入层
                # 浅层投影层（核心）
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

                # 同时支持「编码器前缀」和「交叉注意力前缀」
                self.use_encoder_prefix = True # 使用编码器前缀：适配 Encoder-Decoder 模型，如 T5
                self.use_cross_prefix = True # 使用交叉注意力前缀：适配 Encoder-Decoder 模型的交叉注意力层
                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd) # 编码器前缀嵌入层
                    # 编码器浅层投影层
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                    )

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd) # 交叉注意力前缀投影层
                    # 交叉注意力浅层投影层
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                    )
            # 普通深层参数化（增强表达能力）
            # 与浅层参数化的区别是「投影层增加一层线性 + Tanh」（深层），提升前缀的表达能力，适合复杂任务，其余逻辑与浅层一致
            else:
                low_data_init = 0 # 非低数据场景
                print("UNDER PARAMETRIZATION DEEP 1")

                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                # 深层投影层（多一层 Linear+Tanh）
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

                # 编码器前缀 + 交叉注意力前缀（深层投影层）
                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                    )

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                    )
        # 1. 前缀 dropout 层（防止前缀过拟合，仅作用于前缀参数）
        self.dropout = nn.Dropout(self.prefix_dropout)
        # 2. 中缀场景绑定对应的 forward 方法
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        # 3. 统计可训练参数量（体现 Prefix Tuning 轻量性）
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print("total param is {}".format(total_param))

        # 4. 低数据场景补充初始化（根据 low_data_init 执行对应的训练初始化方法）
        if low_data_init == 2:
            self.lowdata_init_need_tokenize_train(
                gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input
            )
        elif low_data_init == 3:
            print("use pt for this tensor", torch.LongTensor(self.lowdata_token))
            self.lowdata_init_not_need_tokenize_train(
                gpt2=model_gpt2, sample_input=torch.LongTensor(self.lowdata_token)
            )

    def lowdata_init_train1(
        self,
        gpt2: GPT2PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sample_input: str,
    ) -> Parameter:
        """
        用少量样本触发 GPT-2 生成中间特征（past_key_values），并将这些特征转换为可训练参数

        Args:
            gpt2: gpt2 模型
            tokenizer:  与 GPT-2 配套的分词器（transformers.GPT2Tokenizer），用于将文本转换为模型可识别的张量。
            sample_input: 样本输入文本（字符串），用于触发 GPT-2 生成缓存键值对（仅需少量样本即可）。

        Notes:
            tokenizer:
                GPT2Tokenizer 继承自 transformers.PreTrainedTokenizer（所有 Hugging Face 分词器的基类），
                因此它具备所有预训练分词器的核心方法（如 __call__、encode、decode、pad 等）。

            gpt2:
                - GPT2PreTrainedModel 是顶层基类: 处理配置、加载 / 保存权重、设备迁移;
                - GPT2Model 基础编码器（编码器）: 功能：接收输入，输出各层隐藏态;
                                               输出：最后一层隐藏态、可选所有层隐藏态/注意力权重
                                               典型用途：特征提取、下游任务微调（如分类）
                - GPT2LMHeadModel 是带语言建模头的任务模型: 功能：基于 GPT2Model + 语言建模头
                                                        输出：每个 token 的下一个 token 预测概率
                                                        典型用途：文本生成、自回归语言建模（LM 任务）
            应用场景：
            - 低数据量微调：当训练数据极少时，直接用随机初始化的参数训练容易过拟合，用预训练模型的缓存特征初始化参数，可快速收敛。
            - 迁移学习：将 GPT-2 学到的语言特征（蕴含在 key/value 中）作为下游任务的初始化参数，提升模型起点。
            - 参数高效微调（PEFT）：仅训练这部分由缓存初始化的参数，冻结原始 GPT-2 模型权重，减少计算量。
            潜在注意点
            - 输入长度影响：sample_input 的长度会决定 past_key_values 中 seq_len 的维度，需根据下游任务需求选择合适的样本长度。
            - 设备一致性：需确保 sample_input 编码后的张量与 gpt2 模型在同一设备（CPU/GPU），否则会报错【代码中通过 .to(gpt2.device) 处理】。
            - 模型类型：仅适用于支持 past_key_values 的自回归模型（如 GPT 系列），其他模型（如 BERT）不支持该参数。
            - 张量形状：拼接后的张量维度较高（num_layers * 2, 1, num_heads, seq_len, head_dim），需确保后续模块能处理该形状。

        Returns:

        """
        # 使用分词器 tokenizer 处理输入文本 sample_input，将其转换为模型所需的 PyTorch 张量
        # - return_tensors="pt" 指定返回 PyTorch 张量。
        #
        # - 输出 _input 是一个字典，核心键值对：
        #   - input_ids：文本对应的整数编码（每个词/子词映射为一个唯一整数），形状为 [1, seq_len]（1 是批次大小，seq_len 是输入文本的长度）。
        #   - attention_mask（默认自动生成）：注意力掩码，标记哪些位置是有效输入（1）、哪些是填充（0），形状同 input_ids。
        _input = tokenizer(sample_input, return_tensors="pt")

        # 将编码后的 input_ids 传入 GPT-2 模型，执行一次前向推理，核心目的是获取 past_key_values（缓存的键值对）
        #
        # - input["input_ids"].to(gpt2.device)：
        #     将 input_ids 张量移到模型所在设备（CPU/GPU，确保数据和模型在同一设备）
        # - return_dict=True：
        #     指定模型输出为 ModelOutput 对象（类似字典，可通过属性访问结果），而非元组。
        # - use_cache=True：【use_cache=True 仅在推理时有用，训练时通常设为 False（因为需要完整反向传播）。】
        #     核心参数，GPT-2 是自回归模型，use_cache=True 会让模型在推理时缓存每一层 Transformer 的「键」和「值」张量（用于后续快速生成下一个 token），
        #     这些缓存就是 past_key_values。
        # - 输出 output：
        #     包含模型前向结果的对象
        output = gpt2(
            _input["input_ids"].to(gpt2.device), return_dict=True, use_cache=True
        )
        # past_key_values：KV 缓存（Key-Value Cache），用于加速自回归生成（因为 use_cache=True）
        #   结构：(layer0_kv, layer1_kv, ..., layerN_kv)，每层包含 (key, value) 缓存的键值对；即：
        #     Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        #   past_key_values 是一个元组，元组长度为 GPT-2 的 Transformer 层数（比如 GPT-2 基础版有 12 层，元组长度就是 12）
        #     元组中每个元素是一个二元组 (key, value)：
        #   - key：当前层多头注意力的「键」张量，形状为 [1, num_heads, seq_len, head_dim]
        #       （1 = 批次，num_heads = 注意力头数，seq_len = 输入长度，head_dim = 每个头的维度）。
        #   - value：当前层多头注意力的「值」张量，形状与 key 完全一致。
        output = output.past_key_values

        # output[0].shape：
        #   第一层 (key, value) 的形状（即，二元组 (key, value) 的形状，会显示为两个张量的形状）。
        #   比如：对于 GPT-2 基础版，输入长度为 10 时，输出为：
        #       12, (torch.Size([1, 12, 10, 64]), torch.Size([1, 12, 10, 64]))
        #         - 12：Transformer 层数
        #         - 第一个张量 key 的形状（1 = 批次，12 = 注意力头数，10 = 输入长度，64 = 每个头的维度）。
        #         - 第二个张量 value 的形状（与 key 一致）。
        print(len(output), output[0].shape)

        # - torch.cat(output, dim=0)：
        #     将 past_key_values 元组中的所有 (key, value) 二元组沿 dim=0（层数维度）拼接。
        #     拼接前：每个元素是 (key, value)（形状均为 [1, num_heads, seq_len, head_dim]），元组长度为 num_layers。
        #     拼接后：张量形状为 [num_layers * 2, 1, num_heads, seq_len, head_dim]（因为每个层贡献 2 个张量（key+value），所以第一维是 num_layers * 2）。
        # - detach()：
        #     剥离张量的计算图（切断与模型前向传播的梯度依赖），使其成为「纯数据张量」（
        #     后续转换为参数时，梯度会重新计算，此处仅为了脱离原始模型的计算图）。
        output = torch.cat(output, dim=0).detach()

        # 将处理后的张量转换为 torch.nn.Parameter（PyTorch 中的可训练参数）并返回。
        # Parameter 是张量的子类，会被自动注册到模型的参数列表中，
        # 后续训练时会随反向传播更新梯度（这是实现「基于缓存初始化可训练参数」的核心步骤）
        return torch.nn.Parameter(output)

    def get_prompt_p22(self, control_code=None, gpt2=None, bsz: int = None):
        """
        其目的是生成或准备一组可作为 GPT-2 的 past_key_values 的张量，
        通常用于 prompt tuning / prefix tuning / soft prompting 等参数高效微调方法中。

        Args:
            control_code:
            gpt2:
            bsz: 批量大小

        Returns:

        """
        assert bsz is not None
        # - self.control_trans 很可能是一个可学习的 prefix/prompt 参数张量
        #     其形状通常为：(num_layers, bsz, num_heads, prefix_len, head_dim)
        # - 在 Hugging Face 的 GPT-2 中，past_key_values 的标准格式是：
        #     Tuple[Tuple[torch.Tensor, torch.Tensor], ...]  # 长度 = num_layers
        #     每层: (key: [B, H, L, D], value: [B, H, L, D]) 【Batch, num_Heads, Length, Dim】
        # - split(2, dim=0)
        #     self.control_trans 第 0 维长度是 2 * num_layers（比如 24 层 → 48），那么：
        #     返回的是 tuple of 24 tensors，每个 tensor 包含 [key_part, value_part]（堆叠在一起）
        past_key_values = self.control_trans.expand(-1, bsz, -1, -1, -1).split(2, dim=0)
        return past_key_values

    def lowdata_init_need_tokenize_train(
        self, gpt2, tokenizer, sample_input: str, epochs: int = 500
    ) -> None:  # prev=500
        """
        样本训练前使用传入的分词器分词
        Prompt 初始化训练：训练完成后，control_trans 生成的 our_prompt 可作为初始化 Prompt，
        用于下游任务（如文本生成、微调），提升模型在低数据量下的性能。

        用少量样本输入 GPT-2 模型得到「注意力缓存键值对（past_key_values）」，
        以该缓存为「目标值」，通过 MSE 损失训练自定义的 control_trans 模块生成匹配的 Prompt（our_prompt），
        本质是「用预训练模型的中间特征监督 Prompt 生成」，适用于低数据量场景的 Prompt 初始化

        目标：
        在低数据量场景下，避免随机初始化 Prompt 导致的训练不稳定 / 收敛慢问题 ——
        通过 GPT-2 预训练模型的「中间特征（past_key_values）」作为监督信号，
        训练 control_trans 模块生成「符合预训练模型语义分布」的 Prompt，
        为后续下游任务（如文本生成、微调）提供高质量初始化。

        Args:
            gpt2:
            tokenizer:
            sample_input:
            epochs:

        Returns:

        """
        # 自动检测可用设备（优先 GPU，无则用 CPU）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        gpt2 = gpt2.to(device)
        # 生成 GPT-2 目标特征（无梯度追踪）
        # 生成监督目标
        # torch.no_grad() 表示不追踪梯度（仅用 GPT-2 生成特征，不更新 GPT-2 权重）：
        with torch.no_grad():
            # 将文本 sample_input 编码为 GPT-2 可识别的张量, input_ids 形状为 [1, seq_len]
            _input = tokenizer(sample_input, return_tensors="pt")
            # 开启 use_cache=True 以获取 past_key_values（注意力缓存键值对）
            output = gpt2(
                _input["input_ids"].to(gpt2.device), return_dict=True, use_cache=True
            )
            ## !! WARN !! output = output.past_key_values 是「二元组的元组」，直接 cat 会报错
            # past_kv = output.past_key_values
            # # 拆解 (k,v) 二元组，提取所有张量组成纯列表
            # all_kv_tensors = [tensor for k, v in past_kv for tensor in (k, v)]
            # # 拼接为目标特征张量（形状：[num_layers*2, batch_size, num_heads, seq_len, head_dim]）
            # output=torch.cat(all_kv_tensors, dim=0)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        # 创建 Adam 优化器，仅优化 self.control_trans 模块的参数
        # control_trans 是生成 Prompt 的核心模块，自定义实现
        # 学习率：lr=1e-4（适中的学习率，适合 Prompt 微调）
        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        # 均方误差，衡量两个张量的数值差异
        loss_metrics = nn.MSELoss()
        # Prompt 训练循环
        # 每轮迭代的目标：
        #   让 control_trans 生成的 our_prompt 与 GPT-2 的 past_key_values 特征尽可能接近（MSE 损失最小化）：
        for e in range(epochs):
            # 清空梯度（避免梯度累积，每轮迭代独立计算）
            # 由于 optimizer_temp 仅绑定了 self.control_trans 的参数，
            # 也可以调用优化器的 zero_grad() 方法清空梯度（效果与 self.control_trans.zero_grad() 完全一致，更通用）：
            #     optimizer_temp.zero_grad()  # 优化器清空绑定参数的梯度
            self.control_trans.zero_grad()

            # 1、调用自定义方法生成 Prompt（需确保返回格式可拼接，形状匹配 target_feature）
            # get_prompt_p5 返回与 past_key_values 结构匹配的张量序列： 与 past_key_values 一致，或直接返回可拼接的张量列表。
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)

            # 2、计算损失（!! INFO !! 确保 our_prompt 与 output 形状完全一致，否则损失计算报错）
            # 计算 our_prompt 与 output（GPT-2 目标特征）的损失。需确保两者形状完全一致，否则报错
            #   假设使用 gpt2-medium（24 层、16 头、head_dim=64），sample_input 编码后 seq_len=10：
            #   拆解并拼接后的 output 形状：(48, 1, 16, 10, 64)（24 层 × 2 个张量 / 层 = 48，后续维度与 k/v 一致）。
            #   our_prompt 拼接后必须完全匹配该形状（(48, 1, 16, 10, 64)）—— 需确保 get_prompt_p5 生成的 Prompt 张量序列，拼接后维度与目标一致。
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            # 反向传播计算梯度（仅更新 control_trans 的参数，GPT-2 权重固定）。

            # 3、计算梯度，梯度存储在参数的 .grad 属性中
            loss.backward()
            # 可选：梯度裁剪（防止梯度爆炸）
            # torch.nn.utils.clip_grad_norm_(
            #     self.control_trans.parameters(), max_norm=1.0
            # )

            # 4、优化器根据梯度更新参数
            optimizer_temp.step()

            # 5、打印损失（每 10 轮打印一次，避免刷屏）
            if (e + 1) % 10 == 0:
                print(f"Epoch [{e+1}/{epochs}], Loss: {loss.item():.4f}")
        return

    def lowdata_init_not_need_tokenize_train(
        self, gpt2, sample_input: Tensor, epochs=500
    ):  # prev=500
        """
        样本已分词，不需要在方法内再分词
        Args:
            gpt2:
            sample_input: 已分词的训练样本
            epochs:

        Returns:

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        gpt2 = gpt2.to(device)
        with torch.no_grad():
            output = gpt2(
                sample_input.to(gpt2.device), return_dict=True, use_cache=True
            )
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()
        return

    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        """
        用于 生成「结构化 Prompt 张量」的方法：
            将 control_trans 模块的输出转换为与 GPT-2 模型 past_key_values 结构完全匹配的格式，
            以便后续与模型的注意力缓存键值对进行拼接或损失计算
        Args:
            control_code:
            gpt2:
            bsz:

        Returns:

        Notes:
            方法作用：
              - 格式对齐：生成的 past_key_values 与 GPT-2 模型输出的 past_key_values 结构完全一致，可直接用于：
                  - 计算 MSE 损失（如你之前的训练代码中 loss_metrics(our_prompt, output)）。
                  - 与 GPT-2 的注意力缓存拼接（扩展模型输入序列）。
              - 可训练性：self.control_trans 是可训练参数，通过调整它的输出，可让生成的 Prompt 逼近目标特征（如 GPT-2 预训练特征）。
              - 批次适配：通过 expand(bsz, ...) 支持批量处理，提升训练和推理效率。
            总结：
              get_prompt_p2 是一个「格式转换桥梁」，将 control_trans 模块的原始输出转换为与 GPT-2 past_key_values 完全匹配的结构化 Prompt 张量。
                其核心逻辑是：重塑维度 → 扩展批次 → 调整顺序 → 拆分层次，
                最终输出可直接用于与 GPT-2 注意力缓存进行损失计算或拼接的 Prompt 键值对。
              这种设计在「Prompt 微调」或「低数据量初始化」场景中非常关键，确保了生成的 Prompt 能被 GPT-2 模型有效利用。

        """
        assert bsz is not None
        # self.control_trans：生成 Prompt 基础特征的模块（通常是一个可训练的参数或神经网络层，输出原始特征张量）。
        # 1. 重塑 control_trans 输出为 5 维特征；
        #    这一步是为了让 control_trans 的输出维度与 GPT-2 的 key/value 张量维度对齐
        temp_control = self.control_trans.view(
            1,
            self.preseqlen,
            # 每个 Transformer 层包含 key 和 value 两个张量，因此总数量是 层数×2（如 24 层 → 48）
            self.match_n_layer * 2,
            self.match_n_head,  # 注意力头数（与 GPT-2 一致）
            self.match_n_embd,  # 每个注意力头的维度（与 GPT-2 一致）
        ).expand(bsz, -1, -1, -1, -1)
        # 应用 dropout 防止过拟合: 对扩展后的 Prompt 特征进行随机失活（按一定概率将部分元素置为 0），增强模型泛化能力。
        temp_control = self.dropout(temp_control)
        # - permute：调整维度顺序并拆分层次，目的是让输出格式与 GPT-2 的 past_key_values 完全一致,即：
        #   (总key/value数, 批次, 头数, 序列长, 头维度)。
        #   - 原维度索引：[0:bsz, 1:preseqlen, 2:layer×2, 3:head, 4:embd]
        #   - 新维度索引：(match_n_layer×2, bsz, match_n_head, preseqlen, match_n_embd)
        # - split(2) → 按「每 2 个元素一组」拆分张量：
        #   - match_n_layer×2 维度被拆分为 match_n_layer 组，每组包含 2 个张量（分别对应 key 和 value）。
        #   - 拆分后 past_key_values 是一个元组，长度为 match_n_layer（与 GPT-2 层数一致）。
        #   - 元组中每个元素是一个二元组 (key_prompt, value_prompt)，形状均为：
        #     (bsz, match_n_head, preseqlen, match_n_embd)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        # 输出结构：（与 GPT-2 的 past_key_values 完全匹配）
        #   假设 match_n_layer=2（简化为 2 层）、bsz=1、preseqlen=10、match_n_head=16、match_n_embd=64：
        #     - past_key_values 是长度为 2 的元组：
        #         (
        #             (key_prompt_1, value_prompt_1),  # 第 1 层的 Prompt 键值对
        #             (key_prompt_2, value_prompt_2)   # 第 2 层的 Prompt 键值对
        #         )
        #        其中每个 key_prompt_i 和 value_prompt_i 的形状为 (1, 16, 10, 64)，
        #          与 GPT-2 对应层的 key/value 张量形状完全一致。
        return past_key_values

    def get_prompt_p3_infix(
        self, src: Tensor, control_code=None, gpt2=None, bsz=None
    ) -> List[Tensor]:
        """
        生成「插入式（infix）Prompt」
        核心逻辑是：将经过 GPT-2 编码后的输入文本 src 的特征与 control_trans 模块生成的 Prompt 特征拼接，
          形成扩展的注意力缓存键值对（past_key_values）。
        适用于在原有文本特征中间插入自定义 Prompt 的场景（如文本编辑、条件生成等）。

        Args:
            src: 输入文本的 token id
            control_code:
            gpt2:
            bsz: 批次大小

        Returns:
            List[Tensor]: 输入与promt拼接后的Tensor的列表

        Notes:
            - elf.control_trans：
                用于生成 Prompt 特征的模块（输入 src_repr，输出与 GPT-2 注意力缓存兼容的特征）。
            - self.match_n_layer / self.match_n_head / self.match_n_embd：
                与 GPT-2 匹配的层数、注意力头数、每个头的维度（确保 Prompt 与模型结构兼容）。
            - self.dropout：dropout 层（防止过拟合）。

            「infix」意为「插入式」，指该方法生成的 Prompt 不是独立于输入文本的，
               而是与输入文本的特征拼接融合，形成扩展的注意力缓存。
            这种设计适用于：
                - 文本编辑：在原有文本中间插入 Prompt 引导修改（如改写句子、补充内容）。
                - 条件生成：基于输入文本的语义特征生成相关 Prompt，使输出更贴合输入上下文。
                - 特征增强：通过 control_trans 学习输入文本的模式，生成互补的 Prompt 特征，提升模型对输入的理解。
            总结
                get_prompt_p3_infix 的核心逻辑是「输入编码→特征生成→维度对齐→拼接融合」，
                  最终输出包含输入文本与 Prompt 联合特征的注意力缓存。
                与 【get_prompt_p2】 相比，它更强调 Prompt 与输入文本的语义关联（通过 src_repr 驱动 control_trans 生成 Prompt），
                  适用于需要结合输入内容动态生成 Prompt 的场景
        """

        # temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        # print('infix')
        # 将输入文本的 token id 传入 GPT-2，获取编码后的特征。
        # - use_cache=True：返回 past_key_values（输入文本的注意力缓存键值对，用于后续拼接）。
        # - output_hidden_states=True：返回所有层的隐藏态（用于提取输入文本的高层语义特征）。
        src_out = gpt2(
            input_ids=src, use_cache=True, return_dict=True, output_hidden_states=True
        )

        # src_out 包含的核心字段：
        # - hidden_states：所有层的隐藏态（列表）：
        #     - hidden_states[-1] ：最后一层的输出（高层语义特征），作为输入文本的语义表示（src_repr）。
        # - past_key_values：输入文本的注意力缓存（嵌套二元组，形状与之前解析的一致）。
        src_repr = src_out.hidden_states[-1]  # 形状：[bsz, src_seqlen, hidden_size]

        # src_past_key_vals：输入文本 src 经过 GPT-2 各层后的注意力键值对
        #   嵌套二元组，每层为 (key_src, value_src)
        src_past_key_vals = src_out.past_key_values

        # control_trans 模块，生成 Prompt 的原始特征（past_key_values），此时形状为： [bsz, src_seqlen, layer * emb]
        #   layer * emb 是 match_n_layer × 2 × match_n_head × match_n_embd 的扁平化维度。
        # 将输入文本的语义特征 src_repr 传入
        past_key_values = self.control_trans(src_repr)  # [bsz, src_seqlen, layer * emb]

        bsz, seqlen, _ = past_key_values.shape

        # 将 control_trans 输出的扁平特征重塑为与 get_prompt_p2 一致，目的是与 GPT-2 的 key/value 张量维度对齐。
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)  # 随机失活，防止过拟合

        # permute：形状调整为[match_n_layer * 2, bsz, match_n_head, seqlen, match_n_embd]，
        #            与 GPT-2 的 key/value 维度顺序一致
        # split(2)：按层数拆分，得到长度为 match_n_layer 的元组，
        #            每个元素是 (key_prompt, value_prompt) 二元组（Prompt 的键值对）。
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # 拼接输入文本特征与 Prompt 特征
        # 将输入文本的注意力键值对与 Prompt 的键值对沿序列长度维度（dim=3）拼接。
        # full_lst 是长度为 match_n_layer 的元组
        full_lst = []
        for i in range(len(src_past_key_vals)):
            full_lst.append(
                torch.cat([src_past_key_vals[i], past_key_values[i]], dim=3)
            )
        # 输出结构（拼接后的注意力缓存）
        #   假设 GPT-2 为 2 层（match_n_layer=2），bsz=1，src_seqlen=5，match_n_head=16，match_n_embd=64：
        #   full_lst 是长度为 2 的元组：
        #     (
        #         (key_full_1, value_full_1),  # 第 1 层拼接后的键值对
        #         (key_full_2, value_full_2)   # 第 2 层拼接后的键值对
        #     )
        #   每个 key_full_i 和 value_full_i 的形状为 (1, 16, 10, 64)（5 + 5 = 10，序列长度翻倍），
        #     包含输入文本和 Prompt 的联合特征。
        return full_lst

    def get_prompt_p3(
        self, control_code, gpt2=None, bsz=None
    ) -> Tuple[Tuple[Tensor, Tensor]]:
        """
        基于控制码（control_code）生成结构化 Prompt
        将用户输入的 control_code（控制信号，通常是离散的标签或指令编码）通过词嵌入层（wte）转换为向量，
          再经 control_trans 模块处理，最终生成与 GPT-2 模型 past_key_values 结构匹配的 Prompt 键值对。
        适用于条件生成场景（如根据不同控制指令生成特定风格 / 内容的文本）。
        Args:
            control_code:
            gpt2:
            bsz:

        Returns:
            Tuple[Tuple[Tensor, Tensor]]: 元组的元组，元素为(key, value)

        Notes:
            作用: 条件 Prompt 生成
              该方法的核心是将离散的控制信号（control_code）转换为与 GPT-2 兼容的结构化 Prompt，
                实现「控制码 → Prompt → 模型输出」的条件生成链路，
            适用于：
              - 风格控制：用 control_code 表示风格标签（如「正式」「幽默」），生成对应风格的文本。
              - 任务引导：用 control_code 表示任务指令（如「翻译」「摘要」），引导模型执行特定任务。
              - 领域适配：用 control_code 表示领域标签（如「医疗」「法律」），使生成内容贴合特定领域。
            总结
              get_prompt_p3 通过「控制码嵌入 → 特征转换 → 维度对齐」的流程，生成由 control_code 驱动的结构化 Prompt，
                其输出与 GPT-2 的 past_key_values 完全兼容，可直接用于条件生成或微调。
              与 get_prompt_p2（无外部条件）、get_prompt_p3_infix（基于输入文本）相比，
                它更强调外部控制信号对 Prompt 的直接引导，是实现可控生成的重要工具。

        """
        # - self.wte 可选的词嵌入层（Word Token Embedding），
        #   用于将 control_code 转换为向量（若为 None，则使用 GPT-2 自带的 wte）
        # - control_code：控制码张量（形状 [bsz, control_seqlen]），
        #   是生成 Prompt 的条件信号（如标签、指令的 input_ids），不可为 None（否则触发断言错误）。
        # !!! IFNO !!! 将离散的 control_code（整数索引）通过词嵌入层转换为连续向量 temp_control，捕捉控制码的语义信息。
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                # 使用 GPT-2 自带词嵌入
                #   形状为：[bsz, control_seqlen, emb_size]（emb_size 为 GPT-2 隐藏层维度，如 1024）
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # self.control_trans：输入嵌入向量，输出与 GPT-2 注意力缓存兼容的特征。
            # control_trans 模块对嵌入向量 temp_control 进行处理，输出扁平化的 Prompt 特征。
            # 维度说明：layer * emb 是 match_n_layer * 2 * match_n_head * match_n_embd 的乘积（扁平化的键值对特征）。
            past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape

            # self.match_n_layer / self.match_n_head / self.match_n_embd：
            #   与 GPT-2 匹配的层数、注意力头数、头维度（确保 Prompt 结构兼容）。
            past_key_values = past_key_values.view(
                bsz,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
        return past_key_values

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        """
        用于 Prompt 生成
        生成包含「自注意力缓存」、「交叉注意力缓存」等多类型结构的 Prompt 字典，
          适配支持 encoder-decoder 结构或交叉注意力机制的模型（如某些基于 GPT-2 扩展的模型）。
          其输出不仅包含与 GPT-2 past_key_values 兼容的键值对，还可根据配置添加交叉注意力、编码器相关的缓存，
          适用于更复杂的条件生成场景（如文本摘要、翻译等需要编码器-解码器交互的任务）。

       control_trans 是一个 MLP 或线性层，用于将 soft prompt embeddings（temp_control）映射为多层 Transformer 的 Key/Value 缓存格式。
         这是 Prefix Tuning v2 / P-Tuning v2 的典型设计：不直接优化 KV cache，而是通过一个小网络生成它，提升训练稳定性。
         因此，训练目标是让这个小网络学会生成与真实输入（sample_input）对应的 past key-values，从而让模型“以为”看到了该输入

        Args:
            control_code:
            gpt2:
            bsz:
            sample_size: 每个样本的采样数（用于计算扩展后的批次大小，如 sample_size=2 表示每个样本生成 2 个 Prompt）。

        Returns:
            返回「与 past_key_values 结构匹配的张量序列」：与 past_key_values 一致，或直接返回可拼接的张量列表。

        Notes:
            核心作用与适用场景：
              复杂的 Prompt 生成方法，支持多类型注意力缓存，核心作用是：
              - 适配复杂模型结构：不仅支持 GPT-2 式的自回归模型，
                  还兼容带交叉注意力、编码器的模型（如 T5、BART 等 encoder-decoder 模型的变体）。
              - 多维度控制：通过自注意力、交叉注意力、编码器缓存的分别设计，
                  可从多个维度（解码器自身、解码器 - 编码器交互、编码器自身）引导模型生成。
              - 批量采样支持：通过 sample_size 扩展批次，一次生成多个样本的 Prompt，
                  适用于需要多样化输出的场景（如生成多个候选文本）。
            适用场景包括：
              - 文本摘要（需编码器理解输入，解码器生成摘要）
              - 机器翻译（跨语言交叉注意力）
              - 多候选生成（结合 sample_size 生成多个版本）等。
            总结：
              通过「基础自注意力 Prompt + 可选交叉注意力 Prompt + 可选编码器 Prompt」的组合，
                生成结构化的多层缓存字典，适配复杂模型架构和多维度控制需求。
                其设计灵活，通过 use_cross_prefix 和 use_encoder_prefix 开关可适配不同场景，是支持复杂条件生成任务的核心方法。
                输出的字典结构可直接作为模型的 past_key_values 缓存使用，实现 Prompt 对模型生成过程的精细控制。

        """
        old_bsz = bsz
        # 扩展后的批次大小：基础批次 × 采样数
        bsz = bsz * sample_size

        # 1. 生成输入 token 并扩展批次
        # self.input_tokens：虚拟 Prompt token（整数索引张量，作为生成 Prompt 的基础）
        # 形状：[bsz, preseqlen]（preseqlen 是 self.input_tokens 的长度，即 Prompt 序列长）
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)

        # 2. wte 为前缀嵌入
        temp_control = self.wte(
            input_tokens
        )  # temp_control 形状：[bsz, preseqlen, emb_size]

        # 3. 生成自注意力 Prompt 特征
        # self.control_trans / self.control_trans2 / self.control_trans_enc：
        #   对应的特征转换模块，生成不同类型的 Prompt 特征
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape

        # 4. 重塑为 5 维结构（与模型注意力缓存对齐）
        # 形状：[bsz, preseqlen, layer×2, head, head_dim]
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)

        # 5. 按层拆分 (key, value)：生成自注意力机制所需的 Prompt 键值对（key_self/value_self）。
        # 结果为：长度为 match_n_layer 的元组，每层是 (key_self, value_self) 二元组
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # 是否启用交叉注意力 Prompt
        if self.use_cross_prefix:
            # 1. 用另一嵌入层生成输入 token 的向量
            temp_control2 = self.wte2(input_tokens)  # 形状：[bsz, preseqlen, emb_size]

            # 2. 生成交叉注意力 Prompt 特征
            # 形状：bsz, seqlen, layer * emb
            past_key_values2 = self.control_trans2(temp_control2)
            bsz, seqlen, _ = past_key_values2.shape
            # self.match_n_layer / self.match_n_head / self.match_n_embd：
            #   与模型匹配的层数、注意力头数、头维度
            past_key_values2 = past_key_values2.view(
                bsz,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values2 = self.dropout(past_key_values2)
            # 结果：每层是 (key_cross, value_cross) 二元组（交叉注意力键值对）
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        # 是否启用编码器 Prompt
        if self.use_encoder_prefix:
            # 1. 生成编码器输入 token（批次为原始批次 old_bsz，非扩展后的 bsz）
            # 形状：[old_bsz, preseqlen]
            input_tokens_enc = (
                self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            )

            # 2. 编码器嵌入层转换
            # 形状：[old_bsz, preseqlen, emb_size]
            temp_control_enc = self.wte_enc(input_tokens_enc)

            # 3. 生成编码器 Prompt 特征：
            #    编码器 Prompt 用于增强编码器对输入的理解，批次大小为原始 old_bsz（与编码器输入批次一致）
            # 形状：bsz, seqlen, layer * emb （批次是 bsz 吗？） [old_bsz, preseqlen, layer*emb]
            past_key_values_enc = self.control_trans_enc(temp_control_enc)
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(
                bsz_enc,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values_enc = self.dropout(past_key_values_enc)

            # 结果：每层是 (key_enc, value_enc) 二元组（编码器自注意力键值对）
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        # 组装最终结果（多层字典结构）
        # result 是长度为 match_n_layer 的列表，每个元素是一个字典，包含对应层的注意力缓存：
        result = []
        for i, key_val in enumerate(past_key_values):
            # 1. 组装自注意力缓存字典
            temp_dict = {
                "self": {  # 自注意力相关缓存
                    "prev_key": key_val[0].contiguous(),  # key_self（确保内存连续）
                    "prev_value": key_val[1].contiguous(),  # value_self
                    # 注意力掩码（全 False 表示无 padding）
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool(),
                    # bsz, preseqlen
                },
            }

            # 2. 若启用交叉注意力，添加交叉缓存
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict["encoder_decoder"] = {  # 交叉注意力相关缓存
                    "prev_key": key_val2[0].contiguous(),  # key_cross
                    "prev_value": key_val2[1].contiguous(),  # value_cross
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val2.device)
                    .bool(),
                }

            # 3. 若启用编码器，添加编码器缓存
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict["encoder"] = {  # 编码器自注意力相关缓存
                    "prev_key": key_val_enc[0].contiguous(),  # key_enc
                    "prev_value": key_val_enc[1].contiguous(),  # value_enc
                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
                }
            result.append(temp_dict)  # 按层添加到结果列表

        return result

    def get_prompt_p6(self, control_code=None, gpt2=None, bsz=None):
        """
        适用场景：
           - Prompt 嵌入预优化：若 self.input_embs 是通过其他方式（如无监督学习、迁移学习）预优化的向量，
               可直接用于生成高质量 Prompt，避免从头训练嵌入层。
           - 轻量 Prompt 微调：结构简洁，仅通过 control_trans 调整特征，适合模型资源有限、需要快速训练的场景。
        Args:
            control_code:
            gpt2:
            bsz:

        Returns:

        Notes:
            与其他 Prompt 生成方法的对比（核心差异）
              - get_prompt_p2	输入：离散 token（需 view 重塑）；
                                特点：无嵌入层，直接重塑参数；
                                场景：简单 Prompt 初始化
              - get_prompt_p3	输入：控制码 control_code；
                                特点：需词嵌入层（wte），支持条件控制；
                                场景：可控生成（风格、任务引导）
              - get_prompt_p5	输入：离散 token + 多模块；
                                特点：支持自注意力 / 交叉注意力 / 编码器缓存；
                                场景：encoder-decoder 模型、多维度控制
              - get_prompt_p6	输入：预定义嵌入 input_embs；
                                特点：无词嵌入层，简洁高效，直接扩展批次；
                                场景：已优化嵌入的 Prompt、批量生成

        """
        input_embs = self.input_embs.to(self.device)
        # 输出形状：[bsz, preseqlen, layer*emb]
        # 将预定义嵌入 input_embs 传入转换模块，生成扁平化的 Prompt 特征
        #   （layer * emb 是 match_n_layer * 2 * match_n_head * match_n_embd 的乘积，即所有层键值对的扁平化维度）。
        past_key_values = self.control_trans(input_embs).expand(bsz, -1, -1)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        """
        基于控制码（control_code）的 Prompt 生成
        对 control_trans 输出的特征做「序列维度均值池化」—— 通过 mean(1).unsqueeze(1) 压缩控制码的序列信息，
          生成「固定长度为 1」的 Prompt 特征，适用于控制码序列长度不固定，
          但希望 Prompt 长度统一的场景（如用变长标签作为控制信号，生成固定长度的引导 Prompt）。
        Args:
            control_code:
            gpt2:
            bsz:

        Returns:

        Notes:
            设计核心是「用均值池化统一 Prompt 长度」
            适用于以下场景：
              - 控制码序列长度不固定：例如控制码是不同长度的标签（如 “正面”“非常正面”“极度正面”），
                通过均值池化将其压缩为固定长度 1 的 Prompt，避免批次处理时的维度不匹配。
              - 无需依赖控制码序列顺序：当控制码的核心信息是 “类别” 而非 “顺序”（如情感标签、领域标签），
                均值池化能概括整体语义，且简化 Prompt 结构。
              - 轻量化 Prompt 需求：固定长度为 1 的 Prompt 计算开销更小，适合对模型效率有要求的场景。
            不适用场景：
              控制码的序列顺序对 Prompt 生成至关重要（如指令类控制码 “先描述场景，再生成结论”），
                此时均值池化会丢失顺序信息，应使用 get_prompt_p3。
            总结:
              get_prompt_p4 是 get_prompt_p3 的「长度统一版」，
              核心通过 mean(1).unsqueeze(1) 对控制码特征做均值池化，将 Prompt 序列长度固定为 1。
              它继承了 get_p3 「控制码驱动 Prompt 生成」的逻辑，同时解决了控制码序列长度不一致的问题，
              是兼顾灵活性和统一性的轻量化设计，适用于「类别型、变长控制码」的场景。

        """
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                # 自定义嵌入：[bsz, control_seqlen, emb_size]
                # control_seqlen 是控制码的序列长度（可能不固定，如不同标签长度不同）
                # temp_control 形状：[bsz, control_seqlen, emb_size]（批次、控制码长度、嵌入维度）。
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            #
            # - self.control_trans(temp_control) → 形状：[bsz, control_seqlen, layer*emb]
            #     （layer * emb = match_n_layer * 2 * match_n_head * match_n_embd，扁平化特征）
            #
            # - mean(1) → 对「第 1 维（control_seqlen）」求均值，形状：[bsz, layer * emb]
            #      作用：压缩控制码的序列维度，用均值代表整个控制码序列的特征（长度从 control_seqlen → 1）
            #      目的：统一 Prompt 长度，避免因控制码序列长度不一致导致的批次处理问题，
            #           同时用均值概括控制码的整体语义（不依赖具体序列顺序）。
            # - unsqueeze(1) → 在「第 1 维」插入一个维度，形状：[bsz, 1, layer*emb]
            #      作用：恢复序列维度（长度为 1），与后续 reshape 要求的维度匹配
            past_key_values = (
                self.control_trans(temp_control).mean(1).unsqueeze(1)
            )  # bsz, seqlen, layer * emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(
                bsz,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def get_prompt_p1(self, control_code, gpt2=None, bsz=None):
        """
        支持多输入类型、基于控制码的 Prompt 生成方法; 适用于需要用控制码生成固定长度 Prompt 的场景。
        - 区分 control_code 类型（元组 / 非元组），但元组类型已被断言禁用；
        - 非元组场景下，根据 self.embMatch 开关，分「词嵌入 + 求和池化」和「直接使用输入嵌入」两种逻辑；
        - 关键操作是 sum(1).unsqueeze(1) 求和池化（而非 p4 的均值池化），
            且最终会将 Prompt 序列长度扩展为 seq_pastlen × self.preseqlen（固定长度）。
        Args:
            control_code:
            gpt2:
            bsz:
        Notes:
            - self.embMatch (bool)：是否直接使用 control_code 作为嵌入，True 则跳过词嵌入步骤
            - self.wte/gpt2.transformer.wte：词嵌入层（embMatch=False 时用于将离散控制码转为向量）；
            - self.control_trans：特征转换模块（生成与 GPT-2 兼容的扁平化特征）；
            - self.preseqlen：Prompt 的目标序列长度（最终会扩展为该长度）；
            - self.match_n_layer/self.match_n_head/self.match_n_embd：与 GPT-2 匹配的结构参数（层数、头数、头维度）。
            - control_code：必选控制码（非元组时为 input_ids 张量或嵌入向量，元组类型已禁用）；

            求和池化（sum(1)）的特点
            与均值池化（mean(1)）的区别：求和保留了特征的「总强度」（控制码越长、元素值越大，和越大），均值则归一化了长度影响；
            适用场景：当控制码的「语义强度」需要影响 Prompt 特征时（如强控制信号对应强 Prompt 特征），求和池化更合适。

             方法	       计算逻辑	                 特点（影响）	                          适用场景
          sum(dim=1)	序列维度元素求和	   保留长度影响（长序列和更大），体现强度	    控制码长度相近，需保留语义强度
          mean(dim=1)	序列维度元素求均值	   抵消长度影响（长 / 短序列均值范围一致）	控制码长度差异大，需统一尺度
        Returns:



        """

        if control_code is not None:
            if type(control_code) is tuple:  # 直接触发断言错误，禁用元组输入
                assert False, "Tuples"
                control_embs, control_word = control_code
                past_key_values = self.control_trans(control_embs)
                past_key_values = past_key_values.mean(1).unsqueeze(1)
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(
                    bsz,
                    seq_pastlen * self.preseqlen,
                    self.match_n_layer * 2,
                    self.match_n_head,
                    self.match_n_embd,
                )
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                print(control_word, control_embs.shape)
            else:  # control_code 非元组
                # print('running with control code')
                # use the control code to generate the first 5 activation layers.
                if (
                    not self.embMatch
                ):  # 适用于 control_code 是离散 input_ids（整数索引张量）的场景：
                    # 1. 控制码转为词嵌入
                    if self.wte:
                        temp_control = self.wte(
                            control_code
                        )  # 自定义词嵌入：[bsz, control_seqlen, emb_size]
                    else:
                        assert gpt2 is not None
                        temp_control = gpt2.transformer.wte(
                            control_code
                        )  # GPT-2 词嵌入：形状同上
                    # 2. 求和池化（压缩序列维度）
                    # a. sum(1)：对第1维（control_seqlen，控制码序列长度）求和 → 形状：[bsz, emb_size]
                    # b. unsqueeze(1)：插入第1维 → 形状：[bsz, 1, emb_size]（恢复序列维度，长度为1）
                    # 核心：用「求和」概括控制码的全局语义（区别于 p4 的均值，求和保留了特征强度）
                    temp_control = temp_control.sum(1).unsqueeze(1)
                else:  # 直接使用输入的嵌入向量，适用于 control_code 已提前转为嵌入向量（无需词嵌入）的场景
                    # 注：此处未做求和池化，直接将原始嵌入传入后续模块
                    temp_control = control_code  # 形状：[bsz, control_seqlen, emb_size]
                    # print(control_code.shape)

                # 3. 特征转换（生成扁平化 Prompt 特征）
                # - 子逻辑A：[bsz, 1, layer*emb]（layer*emb = match_n_layer×2×match_n_head×match_n_embd）
                # - 子逻辑B：[bsz, control_seqlen, layer*emb]（若 control_seqlen≠1，后续求和会压缩为1）
                past_key_values = self.control_trans(temp_control)
                # print(past_key_values.shape) #bsz, controlCodeLen, long... 5 * config.n_layer * 2 * config.n_embd

                # 4. 求和池化（统一序列长度为1）
                # 形状：[bsz, 1, layer*emb]（无论输入序列长度多少，压缩为1）
                past_key_values = past_key_values.sum(1).unsqueeze(1)
                # print(past_key_values.shape)  # bsz, 1, long...
                bsz, seq_pastlen, _ = past_key_values.shape

                # 形状：[bsz, self.preseqlen, match_n_layer×2, match_n_head, match_n_embd]
                past_key_values = past_key_values.view(
                    bsz,
                    # 1 × self.preseqlen → 固定为 self.preseqlen
                    seq_pastlen * self.preseqlen,
                    self.match_n_layer * 2,
                    self.match_n_head,
                    self.match_n_embd,
                )

                # 6. 维度调整 + 拆分层（与 GPT-2 past_key_values 对齐）
                # 结果：长度为 match_n_layer 的元组，每层是 (key_prompt, value_prompt) 二元组
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
        return past_key_values

    def forward(
        self,
        input_ids=None,
        gpt2_model=None,
        past_key_values=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # labels=None,
        # use_cache=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
    ):
        """
        核心是「Prompt 缓存注入」：通过自定义的 get_prompt 方法生成与 GPT-2 兼容的 Prompt 注意力缓存（past_key_values_prompt），
          将其作为 GPT-2 的初始注意力缓存（past_key_values）传入模型，实现 Prompt 对生成过程的引导。

        Prompt Tuning（提示调优）的核心实现，适用于：
          - 低资源微调：无需微调 GPT-2 全部参数，仅训练 get_prompt 相关模块（如 control_trans、wte），降低计算成本；
          - 可控生成：通过 get_prompt 生成不同 Prompt（如 p3 的控制码驱动、p5 的多类型注意力缓存），引导模型生成特定风格 / 内容的文本；
          - 多场景适配：通过切换 self.get_prompt 方法（如 p1 求和池化、p6 预定义嵌入），适配不同控制码类型、Prompt 长度需求。

        与原生 GPT-2 前向传播的区别：
               对比维度	        原生 GPT-2 前向传播	                     本方法前向传播
          past_key_values	  默认为 None（无初始缓存）	           强制使用 Prompt 生成的初始缓存
          核心控制逻辑	      无 Prompt 引导，仅依赖 input_ids	   Prompt 驱动，可灵活切换 Prompt 类型
          适用场景	          通用生成	                           可控生成、低资源微调

        Args:
            input_ids: 模型输入的 token 索引张量（形状 [bsz, seqlen]），必选（用于推断批次大小 bsz）
            gpt2_model: GPT-2 模型实例（必选，用于执行前向传播）
            past_key_values: 外部传入的注意力缓存（已禁用，强制使用 Prompt 生成的缓存）
            src: 输入序列（预留参数，mode_para=2 时可能用于生成 Prompt 或拼接掩码）
            tgt: 目标序列（预留参数，mode_para=2 时可能用于生成 Prompt 或拼接掩码）
            src_attn: 输入序列的注意力掩码（mode_para=2 时拼接为完整掩码）
            tgt_attn: 目标序列的注意力掩码（mode_para=2 时拼接为完整掩码）
            **kwargs: 兼容 GPT-2 模型其他前向参数（如 labels、output_hidden_states 等）

        Returns:

        Notes:
            - self.get_prompt: Prompt 生成方法
            - self.mode_para: 运行模式开关（mode_para=2 时启用特殊逻辑：拼接注意力掩码）
              mode_para=2 的特殊意义：从注释逻辑推测，mode_para=2 是「输入序列关联模式」：
                此时 get_prompt 会传入 src（输入序列），生成与输入相关的 Prompt（如 p3_infix 插入式 Prompt，拼接输入序列特征）；
                同时拼接 src_attn 和 tgt_attn 掩码，适配「输入 - 目标」双序列场景（如文本摘要、翻译）。
            Prompt 如何影响模型生成？
              GPT-2 的 past_key_values 是「历史注意力缓存」，用于存储之前 token 的 key/value 特征（避免重复计算）。本方法中：
                - Prompt 生成的 past_key_values 会作为「初始历史缓存」传入模型；
                - 模型生成第一个 token 时，会同时关注 input_ids 和 Prompt 缓存的特征，
                  从而被 Prompt 引导（如遵循 Prompt 设定的风格、任务指令）。
            为什么禁用外部 past_key_values？
              - 避免冲突：如果外部传入 past_key_values，会覆盖 Prompt 缓存，导致 Prompt 失效；
              - 设计初衷：本方法的核心是「Prompt 驱动」，强制使用自定义 Prompt 缓存，确保控制逻辑的一致性。
        """

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        # mode_para=2 时，传入 src 生成 Prompt（可能是 infix 插入式 Prompt）
        # if self.mode_para == 2:
        #     past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        # else:

        # 生成与 GPT-2 兼容的 past_key_values 结构（每层 (key_prompt, value_prompt) 二元组）。
        # 两种模式：
        #   - 普通模式（默认）：仅传入 bsz，生成独立的 Prompt 缓存（如 p2/p6，不依赖输入序列 src）；
        #   - mode_para=2 模式（注释中逻辑）：传入 src 和 gpt2_model，生成与输入序列相关的 Prompt（如 p3_infix 插入式 Prompt）。
        past_key_values_prompt = self.get_prompt(bsz=bsz)

        # 禁用外部 past_key_values，强制使用 Prompt 缓存
        # 限制：不允许外部传入 past_key_values（避免与 Prompt 缓存冲突），
        # 强制模型使用 self.get_prompt 生成的 Prompt 缓存作为【初始注意力缓存】。
        # 确保 Prompt 能完全引导模型的注意力计算，避免外部缓存干扰
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        # 当 mode_para=2（可能是 encoder-decoder 或序列拼接场景）时，
        #   将输入序列掩码 src_attn 和目标序列掩码 tgt_attn 沿「序列长度维度（dim=1）」拼接，
        #   形成完整的注意力掩码（避免模型关注 Padding 或无效区域）。
        # 非 mode_para=2 时：不拼接掩码，attention_mask 由 **kwargs 传入或使用模型默认值。
        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)

        # 将「输入 input_ids」和「Prompt 注意力缓存 past_key_values」传入 GPT-2，
        #   模型会在生成时融合 Prompt 的特征（通过注意力机制关注 Prompt 的 key/value）。
        # **kwargs：支持传入 GPT-2 其他参数（如 labels 用于微调损失计算、output_hidden_states=True 输出隐藏态等）。
        # 输出格式：包含 logits、past_key_values、hidden_states 等，可直接用于生成文本或计算微调损失。
        output = gpt2_model(
            input_ids=input_ids, past_key_values=past_key_values, **kwargs
        )

        # output = gpt2_model(input_ids=input_ids,
        #                     past_key_values=past_key_values, attention_mask=attention_mask,
        #                     token_type_ids=token_type_ids, position_ids=position_ids,
        #                    head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
        #                    encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
        #                    output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        #                    return_dict=return_dict, **kwargs)

        return output

    def forward_infix(
        self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
    ):
        """
        针对「插入式 Prompt（Infix Prompt）」设计的前向传播方法，相比之前的 forward 方法，
          它强化了「多序列拼接」和「插入式 Prompt」的适配。
        新增参数为插入式 Prompt 提供更灵活的控制：
          - control_code：驱动 Prompt 内容
          - weights：调整 Prompt 影响强度
        插入式 Prompt 的核心：
          不是作为「前缀」加在输入前，而是作为「中间层」插入到「输入序列 src」和「目标序列 tgt」之间，
            让模型在处理 tgt 时同时关注 src 和 Prompt。
        核心逻辑是：
          将 Prompt 作为「中间插入层」融入模型，通过拼接多段注意力掩码（src_attn + src_attn + tgt_attn）
          适配「输入序列 - Prompt - 目标序列」的三段式结构，实现更精细的 Prompt 引导（区别于普通的「前缀 Prompt」）。

        Args:
            input_ids:
            weights: 用于 Prompt 生成的权重调整
            control_code: 用于 Prompt 生成的控制码
            emb_match: 用于 Prompt 生成的嵌入匹配开关（传递给 GPT-2 或 get_prompt）
            past_key_values:
            attention_mask: 原始注意力掩码（最终会被「三段式掩码」覆盖）
            token_type_ids:
            position_ids:
            head_mask:
            inputs_embeds:
            encoder_hidden_states:
            encoder_attention_mask:
            labels:
            use_cache:
            output_attentions:
            output_hidden_states:
            return_dict:
            gpt2_model:
            src:
            tgt:
            src_attn:
            tgt_attn:
            cate_batch: 类别相关批次（预留参数，可能用于类别特异性 Prompt 生成）
            cate_attn: 类别相关掩码（预留参数，可能用于类别特异性 Prompt 生成）
            **kwargs:

        Returns:

        Notes:
            与普通 forward 方法的核心差异
              对比维度	           普通 forward 方法	                   forward_infix 方法
            Prompt 类型	         前缀 Prompt（加在输入前）	         插入式 Prompt（夹在 src 和 tgt 之间）
            get_prompt 输入	     可选传入 src（仅 mode_para=2）	 强制传入 src（与输入序列关联）
            注意力掩码	         单段掩码（或两段拼接）	             三段式拼接掩码（src_attn + Prompt_attn + tgt_attn）
            控制参数	             无额外控制参数	                 支持 weights/emb_match 等扩展参数
            适用场景	             通用可控生成	                     输入 - Prompt - 目标的三段式场景（如摘要、翻译、文本续写）

        四、设计意图与适用场景
        1. 设计意图
          - 解决「前缀 Prompt」的局限性：前缀 Prompt 仅在输入前引导，
              插入式 Prompt 能在「输入→目标」的过渡中持续引导，更适合需要输入与目标强关联的任务；
          - 语义对齐：通过复用 src 的掩码作为 Prompt 掩码，确保 Prompt 与输入序列 src 语义一致，
              引导模型在生成 tgt 时兼顾 src 和 Prompt 的约束。
        2. 适用场景
          - 文本摘要：src 是原始文本，Prompt 是「生成简洁摘要」的指令，tgt 是摘要文本
             模型在原始文本和摘要之间插入 Prompt，确保摘要贴合原文且符合指令
          - 机器翻译：src 是源语言文本，Prompt 是「翻译成目标语言」的指令，tgt 是目标语言文本
             插入式 Prompt 强化跨语言语义对齐
          - 文本续写：src 是前文，Prompt 是「保持风格一致续写」的指令，tgt 是续写文本
             插入式 Prompt 确保续写内容与前文风格、语义连贯
        五、关键注意事项
          - Prompt 长度与掩码匹配：拼接后的掩码长度必须与
              「src 长度 + Prompt 长度 + tgt 长度」完全一致，否则会导致维度不匹配报错；
          - src_attn 复用逻辑：Prompt 掩码复用 src_attn 意味着 Prompt 与 src 无 Padding 且长度一致，
              若 Prompt 长度与 src 不同，需单独生成 Prompt 的掩码（而非复用 src_attn）；
          - 控制参数的作用：weights/emb_match 等参数需 GPT-2 模型支持（或在 get_prompt 中处理），
              否则可能无效（代码中 control_code 显式置空，说明控制逻辑已在 Prompt 生成阶段完成）。
        总结
          forward_infix 是专为「插入式 Prompt」设计的前向传播方法，
            核心创新是「三段式结构（src-Prompt-tgt）+ 三段式掩码拼接」，让 Prompt 能在输入与目标之间发挥过渡引导作用，
              更适合需要输入 - 目标强关联的任务（如摘要、翻译）。
            相比普通 forward，它强化了与输入序列的语义对齐，提供了更灵活的控制参数，
              是 Prompt Tuning 框架中针对复杂生成任务的进阶实现。
        """

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]
        # 生成插入式 Prompt 缓存
        #
        # 向 get_prompt 传入 src（输入序列）和 gpt2_model，
        #   说明生成的 Prompt 是与输入序列 src 相关的插入式 Prompt（对应之前的 get_prompt_p3_infix 逻辑）。
        #
        past_key_values_prompt = self.get_prompt(
            src, None, gpt2=gpt2_model, bsz=bsz
        )

        # 拼接三段式注意力掩码；目的是：
        #   告诉模型，三段式结构中哪些元素是有效数据（无 Padding），
        #     确保注意力计算只关注有效元素，避免 Prompt 或 Padding 干扰。
        #
        # 掩码拼接逻辑对应「输入序列 src - Prompt - 目标序列 tgt」的三段式结构：
        #   - 第一段 src_attn：输入序列 src 的注意力掩码（标记 src 中的有效元素）；
        #   - 第二段 src_attn：Prompt 的注意力掩码（复用 src 的掩码，说明 Prompt 与 src 语义对齐，无额外 Padding）；
        #   - 第三段 tgt_attn：目标序列 tgt 的注意力掩码（标记 tgt 中的有效元素）；
        #   - 拼接维度 dim=1：沿「序列长度维度」拼接，最终掩码形状为 [bsz, len(src) + len(Prompt) + len(tgt)]，覆盖整个三段式序列。
        attention_mask = torch.cat(
            [src_attn, src_attn, tgt_attn], dim=1
        )  # bsz, seqlen

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"
        # 相比普通 forward，新增了 weights/emb_match 等控制参数，
        #   让模型能根据需求调整 Prompt 的作用方式（如 weights 越大，Prompt 对生成的影响越强）。
        output = gpt2_model(
            input_ids=input_ids,
            control_code=None,   # 显式置空（控制逻辑已在 Prompt 生成时完成）
            weights=weights,     # 传递权重参数（可能用于调整 Prompt 影响强度）
            emb_match=emb_match, # 传递嵌入匹配开关
            past_key_values=past_key_values,  # 插入式 Prompt 缓存
            attention_mask=attention_mask,    # 三段式拼接掩码
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return output


class PrefixEmbTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""

    def __init__(
        self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False
    ):
        super().__init__(config)

        print("under the PrefixEmbTuning model")

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        if hasattr(config, "optim_prefix"):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, "preseqlen") and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, "use_infix"):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, "_my_arg_tune_mode"):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = "prefixtune"

        if hasattr(config, "_my_arg_task_mode"):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = "underspecified"
            assert False, "the task is underspecified"

        if hasattr(config, "train_weights"):
            self.train_weights = config.train_weights == "yes"
        else:
            assert False, "unspecified train weights"

        if hasattr(config, "format_mode"):
            self.format_mode = config.format_mode
        else:
            self.format_mode = "cat"

        if hasattr(config, "prefix_dropout"):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        if hasattr(config, "init_random"):
            self.init_random = config.init_random == "yes"
        else:
            self.init_random = False

        if hasattr(config, "mid_dim"):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        # if hasattr(config, 'mid_layers'):
        #     self.mid_layers = config.mid_layers
        # else:
        #     self.mid_layers = 1

        if False:
            if hasattr(config, "_my_arg_task_mode"):
                self.task_mode = config._my_arg_task_mode
            else:
                self.task_mode = "under-specified"
                print("the task is underspecified")
                assert False

            if hasattr(config, "train_weights"):
                self.train_weights = config.train_weights == "yes"
            else:
                self.train_weights = False
                assert False, "train_weights should be specified."

            print("train embedding is {}".format(self.train_weights))

            if hasattr(config, "_my_arg_control"):
                print("control mode is on.")
                self.prefix_control = True
            else:
                self.prefix_control = False
                assert False, "the control is underspecified"

        if self.task_mode == "dataless":
            self.mode_para = 1
        elif (
            self.task_mode == "data2text"
            or self.task_mode == "triples"
            or self.task_mode == "webnlg"
            or self.task_mode == "writingPrompts"
            or self.task_mode == "summarization"
        ):
            # with src and input based encoding.
            self.mode_para = 2
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4

        if not self.optim_prefix:
            if self.train_weights:
                self.wte = model_gpt2.transformer.wte
                for p in self.wte.parameters():
                    p.requires_grad = True
            else:
                if not self.init_random:
                    self.wte = None
                else:
                    print(
                        "the is just for baseline checking!!! We reinitialize the LM embeddings and try cat "
                        "and peek."
                    )
                    print("BASELINE" * 100)
                    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
                    print(self.wte)

            if self.mode_para == 1:
                print("mode_para=1, for dataless.")
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p4_infix
                else:
                    self.get_prompt = self.get_prompt_p4
            elif self.mode_para == 2 or self.mode_para == 4:
                print(
                    "mode_para=2 or 4, for (2)data2text having a variable length input prefix parametrization. or for (4) topic/keyword/attributes..."
                )

                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p3_infix
                else:
                    self.get_prompt = self.get_prompt_p3

        else:
            self.mode_para = 0
            print(
                "mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) "
            )
            print(
                "preseqlen is {}, under the mode of optimizing prefix directly".format(
                    self.preseqlen
                )
            )

            # DIFFERENT PARAMETRIZATION:
            if True:
                print("UNDER PARAMETRIZATION 1")
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

            # DIFFERENT PARAMETRIZATION 2.
            elif True:
                print("UNDER PARAMETRIZATION 2")
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
                input_word_lst = [
                    [
                        "name",
                        "Type",
                        "price",
                        "customer rating",
                        "near",
                        "area",
                        "family friendly",
                    ]
                ]
                input_word_ids = tokenizer(
                    input_word_lst,
                    add_special_tokens=True,
                    is_split_into_words=True,
                    return_tensors="pt",
                )["input_ids"]
                self.input_embs = model_gpt2.transformer.wte(
                    input_word_ids.to(model_gpt2.device)
                )
                print(self.input_embs.shape)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p6_infix
                else:
                    self.get_prompt = self.get_prompt_p6

            # OLD CODE.
            # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
            # if self.use_infix:
            #     assert False, "just optimizing a set of parameter is not really related to infix position."
            #     self.get_prompt = self.get_prompt_p2_infix
            # else:
            #     self.get_prompt = self.get_prompt_p2

        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print("total param is {}".format(total_param))

        ############################################################################

    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        """
        Directly specifying/optimizing the input embeddings.
        :param control_code:
        :param gpt2:
        :param bsz:
        :return:
        """
        assert bsz is not None
        temp_control = self.control_trans.unsqueeze(0).expand(
            bsz, -1, -1
        )  # bsz, seqlen, emb
        temp_control = self.dropout(temp_control)
        temp_result = gpt2(inputs_embeds=temp_control, use_cache=True)
        past_key_values = temp_result.past_key_values
        return past_key_values

    def get_prompt_p2_infix(self, src_x, control_code=None, gpt2=None, bsz=None):
        """
        Directly specifying/optimizing the input embeddings.
        :param control_code:
        :param gpt2:
        :param bsz:
        :return:
        """
        assert bsz is not None
        temp_control = self.control_trans.unsqueeze(0).expand(
            bsz, -1, -1
        )  # bsz, seqlen, emb
        temp_control = self.dropout(temp_control)
        src_embs = gpt2.wte(src_x)
        print(temp_control.shape, src_embs.shape)
        temp_control = torch.cat([src_embs, temp_control], dim=1)
        print(temp_control.shape)
        temp_result = gpt2(inputs_embeds=temp_control, use_cache=True)
        past_key_values = temp_result.past_key_values
        return past_key_values

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        input_embs = self.control_trans(temp_control)  # bsz, seqlen, emb_dim
        bsz, seqlen, _ = input_embs.shape
        input_embs = self.dropout(input_embs)
        temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        past_key_values = temp_result.past_key_values

        return past_key_values

    def get_prompt_p3_infix(self, src_x, control_code, gpt2=None, bsz=None):
        """
        专为「插入式 Prompt（Infix Prompt）」设计的缓存生成方法，
        核心逻辑是将「输入序列嵌入（src_embs）」与「控制码驱动的 Prompt 嵌入（input_embs）」沿序列维度拼接，
          再通过 GPT-2 前向传播自动生成原生兼容的 past_key_values。
        这种设计让 Prompt 作为「中间插入层」夹在输入序列和后续生成序列之间，实现更精细的语义引导（区别于前缀 Prompt）。
        Args:
            src_x:
            control_code:
            gpt2:
            bsz:

        Returns:

        Notes:
            - temp_control = self.wte(control_code)
            - input_embs = self.control_trans(temp_control) # 作用：优化控制码嵌入，让其更贴合 GPT-2 的语义空间，同时确保维度完全匹配。
            -  temp_result = gpt2( # 作用：让 GPT-2 自动处理 Prompt 嵌入，生成原生兼容的注意力缓存（无需手动构造维度）
                inputs_embeds=input_embs, use_cache=True, return_dict=True
            )
            「控制码 → Prompt 嵌入 → GPT-2 原生缓存生成」的核心链路，
               本质是用控制码驱动生成符合 GPT-2 规范的注意力缓存（past_key_values），无需手动构造维度，兼容性和稳定性拉满。


        """
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb

            src_embs = gpt2.transformer.wte(src_x)
            input_embs = self.control_trans(temp_control)  # bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            input_embs = torch.cat([src_embs, input_embs], dim=1)
            # print(input_embs.shape)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(
                inputs_embeds=input_embs, use_cache=True, return_dict=True
            )
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def get_prompt_p3(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            input_embs = self.control_trans(temp_control)  # bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(
                inputs_embeds=input_embs, use_cache=True, return_dict=True
            )
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        """
        基于 GPT-2 原生前向传播生成 Prompt 注意力缓存（past_key_values）
        不依赖手动维度重塑，而是让 GPT-2 直接处理 Prompt 嵌入，自动生成符合模型规范的 past_key_values，兼容性更强、出错率更低。
        Args:
            control_code:
            gpt2:
            bsz:

        Returns:

        """
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            input_embs = self.control_trans(temp_control)  # bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(
                inputs_embeds=input_embs, use_cache=True, return_dict=True
            )
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def forward_infix(
        self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
    ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]
        # TODO-LISA
        self.format_mode = "cat"
        if self.mode_para == 2:
            if self.format_mode == "cat":
                # 子模式1：用「输入 src + 类别 cate_batch」生成 Prompt
                past_key_values_prompt = self.get_prompt(
                    src, cate_batch, gpt2=gpt2_model, bsz=bsz
                )
                # 三段式掩码：输入掩码 + 类别掩码 + 目标掩码（对应 Prompt 格式）
                attention_mask = torch.cat([src_attn, cate_attn, tgt_attn], dim=1)
            else:
                # 子模式2：用「输入 src + 输入 src」生成 Prompt（兼容早期逻辑）
                past_key_values_prompt = self.get_prompt(
                    src, src, gpt2=gpt2_model, bsz=bsz
                )
                # 三段式掩码：输入掩码 + 输入掩码 + 目标掩码
                attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1)
        else:

            past_key_values_prompt = self.get_prompt(
                src, None, gpt2=gpt2_model, bsz=bsz
            )
            # 自动生成 Prompt 对应的注意力掩码（全 True，无 Padding）
            bsz, seqlen = src.shape
            # 核心亮点：temp_attn 自动生成 —— 无需外部传入 Prompt 掩码，
            #   根据 self.preseqlen（预设 Prompt 长度）生成全 True 掩码，
            #   默认 Prompt 无 Padding（有效长度 = preseqlen）；
            # 适用场景：Prompt 是固定长度的通用指令（如 “生成简洁、正式的文本”），无需类别约束，简化掩码构造流程。
            temp_attn = torch.ones(bsz, self.preseqlen).bool()  # [bsz, preseqlen]，全为 True
            # 三段式掩码：输入掩码 + Prompt 掩码（全 True） + 目标掩码
            attention_mask = torch.cat([src_attn, temp_attn, tgt_attn], dim=1)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        # if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
        #     attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(
            input_ids=input_ids,
            control_code=None,
            weights=weights,
            emb_match=emb_match,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return output

    def forward(
        self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
    ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            past_key_values_prompt = self.get_prompt(
                control_code, gpt2=gpt2_model, bsz=bsz
            )
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(
            input_ids=input_ids,
            control_code=None,
            weights=weights,
            emb_match=emb_match,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return output
