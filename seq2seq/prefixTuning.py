# from transformers import Trainer
import torch
from torch.nn import Embedding, Sequential, Parameter

from transformers import (
    GPT2PreTrainedModel,
    PreTrainedTokenizer,
    GPT2Tokenizer,
    PretrainedBartModel,
)
from torch import nn
from typing import Optional, Literal


class PrefixTuning(PretrainedBartModel):
    """Classification Head for  transformer encoders

    Attributes:
        preseqlen (int): 前缀优化序列长度
        optim_prefix (bool): 是否前缀优化
        use_infix (bool): 是否使用 infix
        use_deep (bool):  是否使用深度模式
        n_layer (int):
        match_n_layer (int): 解码器层数
        match_n_head (int):  解码器注意力头数
        n_embd (int):  嵌入的维度
        task_mode (str):  任务模式
        tuning_mode (str):  微调模式：prefixtune
        train_weights (bool):  是否训练权重
        format_mode (str):  前缀与输入的拼接格式 ["cat", "infix", "peek", "nopeek"]
        prefix_dropout (float): 前缀 dropout 值
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
        mode_para (int): [0, 1(dataless), 2(writingPrompts、webnlg、triples、data2text),3, 4]
        wte (Embedding): 权重嵌入
        wte_enc (Embedding): encoder prefix 的权重嵌入
        control_trans (Sequential): 控制转换，是一个 MLP

    """

    preseqlen: int = 5
    optim_prefix: bool = False
    use_infix: bool = False
    use_deep: bool = False
    n_layer: int
    match_n_layer: int
    match_n_head: int
    n_embd: int
    task_mode: Literal["writingPrompts", "webnlg", "triples", "data2text", "dataless"]
    tuning_mode: str
    train_weights: bool
    format_mode: Literal["cat", "infix", "peek", "nopeek"] = "cat"
    prefix_dropout: float = 0.0
    init_random: bool = False
    mid_dim: int = 512
    lowdata: bool = False
    lowdata_token: Optional[str] = None
    mode_para: Literal[0, 1, 2, 3, 4] = 1
    wte: Embedding
    wte_enc: Embedding
    control_trans: Sequential
    use_cross_prefix: bool
    use_encoder_prefix: bool = True

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

        if hasattr(config, "use_deep"):
            self.use_deep = config.use_deep == "yes"
        else:
            self.use_deep = False

        deep_param = self.use_deep

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

        if self.task_mode == "dataless":
            self.mode_para = 1
        elif (
            self.task_mode == "data2text"
            or self.task_mode == "triples"
            or self.task_mode == "webnlg"
            or self.task_mode == "writingPrompts"
        ):
            # with src and input based encoding.
            self.mode_para = 2
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4
        # 不是前缀优化
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

            # dataless
            if self.mode_para == 1:
                print("mode_para=1, for dataless.")
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
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
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p3_infix
                else:
                    self.get_prompt = self.get_prompt_p3

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
            # 低数据场景，并且没有初始化 token
            if self.lowdata and self.lowdata_token is not None:
                low_data_init = 3
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
                elif low_data_init == 2:
                    print(
                        "IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, need to train first"
                    )
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
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

                elif low_data_init == 3:
                    # use a single prepended token.
                    assert self.lowdata_token is not None
                    self.preseqlen = len(self.lowdata_token[0])
                    print(
                        "IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, low_data_init=3, "
                        "preseqlen = {} Unifying with FINETUNE".format(self.preseqlen)
                    )
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
                    )
                    self.get_prompt = self.get_prompt_p5

            # DIFFERENT PARAMETRIZATION:
            elif not deep_param:
                low_data_init = 0
                print("UNDER PARAMETRIZATION 1")
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                )
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                    )

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                    )

            else:
                low_data_init = 0
                print("UNDER PARAMETRIZATION DEEP 1")

                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
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

        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print("total param is {}".format(total_param))

        if low_data_init == 2:
            self.lowdata_init_train2(
                gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input
            )
        elif low_data_init == 3:
            print("use pt for this tensor", torch.LongTensor(self.lowdata_token))
            self.lowdata_init_train3(
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

    def get_prompt_p22(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        past_key_values = self.control_trans.expand(-1, bsz, -1, -1, -1).split(2, dim=0)
        return past_key_values

    def lowdata_init_train2(
        self, gpt2, tokenizer, sample_input, epochs=500
    ):  # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            input = tokenizer(sample_input, return_tensors="pt")
            output = gpt2(
                input["input_ids"].to(gpt2.device), return_dict=True, use_cache=True
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

    def lowdata_init_train3(self, gpt2, sample_input, epochs=500):  # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
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
        assert bsz is not None
        temp_control = self.control_trans.view(
            1,
            self.preseqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        ).expand(bsz, -1, -1, -1, -1)
        temp_control = self.dropout(temp_control)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p3_infix(self, src, control_code=None, gpt2=None, bsz=None):
        # temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        # print('infix')
        src_out = gpt2(
            input_ids=src, use_cache=True, return_dict=True, output_hidden_states=True
        )
        src_repr = src_out.hidden_states[-1]  # bsz, seqlen, hidden
        src_past_key_vals = src_out.past_key_values
        past_key_values = self.control_trans(src_repr)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        # print(past_key_values.shape)
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        full_lst = []
        for i in range(len(src_past_key_vals)):
            full_lst.append(
                torch.cat([src_past_key_vals[i], past_key_values[i]], dim=3)
            )

        return full_lst

    def get_prompt_p3(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
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
        return past_key_values

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if self.use_cross_prefix:
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(
                temp_control2
            )  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(
                bsz,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        if self.use_encoder_prefix:
            input_tokens_enc = (
                self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            )
            temp_control_enc = self.wte_enc(input_tokens_enc)
            past_key_values_enc = self.control_trans_enc(
                temp_control_enc
            )  # bsz, seqlen, layer*emb
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(
                bsz_enc,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {
                "self": {
                    "prev_key": key_val[0].contiguous(),
                    "prev_value": key_val[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool(),
                    # bsz, preseqlen
                },
            }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict["encoder_decoder"] = {
                    "prev_key": key_val2[0].contiguous(),
                    "prev_value": key_val2[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val2.device)
                    .bool(),
                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict["encoder"] = {
                    "prev_key": key_val_enc[0].contiguous(),
                    "prev_value": key_val_enc[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
                }
            result.append(temp_dict)

        return result

    def get_prompt_p6(self, control_code=None, gpt2=None, bsz=None):
        input_embs = self.input_embs.to(self.device)
        past_key_values = self.control_trans(input_embs).expand(
            bsz, -1, -1
        )  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = (
                self.control_trans(temp_control).mean(1).unsqueeze(1)
            )  # bsz, seqlen, layer*emb
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
        if control_code is not None:

            if type(control_code) is tuple:
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
            else:
                # print('running with control code')
                # use the control code to generate the first 5 activation layers.
                if not self.embMatch:
                    if self.wte:
                        temp_control = self.wte(control_code)
                    else:
                        assert gpt2 is not None
                        temp_control = gpt2.transformer.wte(control_code)
                    temp_control = temp_control.sum(1).unsqueeze(1)
                else:
                    temp_control = control_code
                    # print(control_code.shape)
                past_key_values = self.control_trans(temp_control)
                # print(past_key_values.shape) #bsz, controlCodeLen, long... 5 * config.n_layer * 2 * config.n_embd
                past_key_values = past_key_values.sum(1).unsqueeze(1)
                # print(past_key_values.shape)  # bsz, 1, long...
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(
                    bsz,
                    seq_pastlen * self.preseqlen,
                    self.match_n_layer * 2,
                    self.match_n_head,
                    self.match_n_embd,
                )
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
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

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        # if self.mode_para == 2:
        #     past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        # else:

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)

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

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(
                src, None, gpt2=gpt2_model, bsz=bsz
            )
            attention_mask = torch.cat(
                [src_attn, src_attn, tgt_attn], dim=1
            )  # bsz, seqlen
        else:
            past_key_values_prompt = self.get_prompt(
                src, None, gpt2=gpt2_model, bsz=bsz
            )
            attention_mask = torch.cat(
                [src_attn, src_attn, tgt_attn], dim=1
            )  # bsz, seqlen

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

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
                past_key_values_prompt = self.get_prompt(
                    src, cate_batch, gpt2=gpt2_model, bsz=bsz
                )
                attention_mask = torch.cat([src_attn, cate_attn, tgt_attn], dim=1)
            else:
                past_key_values_prompt = self.get_prompt(
                    src, src, gpt2=gpt2_model, bsz=bsz
                )
                attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1)
        else:

            past_key_values_prompt = self.get_prompt(
                src, None, gpt2=gpt2_model, bsz=bsz
            )
            bsz, seqlen = src.shape
            temp_attn = torch.ones(bsz, self.preseqlen).bool()
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
