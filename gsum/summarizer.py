import copy
import functools
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch import nn
from transformers import AutoModel
from typing import Callable, List, Optional

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule

from .preprocess import TARGET_BOS, TARGET_EOS, TARGET_SEP

#
# Lightning modules
#
class GuidedAbsSum(pl.LightningModule):
    """
    Abstractive summarization based on GSum.
    """

    def __init__(self, cfg: GuidedSummarizationConfig, data_module: GuidedSummarizationDataModule):
        super(GuidedAbsSum, self).__init__()
        self.config = cfg
        self.data_module = data_module

        # Encoder
        self.enc = AbsSumTransformerEncoder(cfg)  # AbsTransformerEncoder

        # Shared embedding layer
        embedding = nn.Embedding(self.enc.bert.config.vocab_size, self.enc.bert.config.hidden_size, padding_idx=0)
        embedding.weight = copy.deepcopy(self.enc.bert.embeddings.word_embeddings.weight)

        # Decoder
        self.dec = AbsSumTransformerDecoder(
            cfg, cfg.decoder_layers, cfg.decoder_dim, cfg.decoder_heads, cfg.decoder_ff_dim, cfg.decoder_dropout,
            embedding, self.enc.bert.config.vocab_size)

        # Generator
        self.gen = nn.Sequential(
            nn.Linear(self.enc.bert.config.hidden_size, self.enc.bert.config.vocab_size),
            nn.LogSoftmax(dim=-1))

        self.loss_func = LabelSmoothingLoss(cfg.label_smoothing, self.enc.bert.config.vocab_size, 0)

        pass
        #
        # TODO: Weights initialization?
        #

    def forward(self, x: List[str]) -> List[str]:
        x_prepared = self.data_module.preprocess(x)
        return [self.infer(i['x_input']) for i in x_prepared]

    def infer(self, x_input: dict) -> str:
        self.enc.to(self.device)
        self.dec.to(self.device)
        self.gen.to(self.device)

        top_vec, gui_vec = self.enc(x_input)
        results: List[BeamSearchResult] = [BeamSearchResult.create(self.data_module.tokenizer.convert_tokens_to_ids(TARGET_BOS), TARGET_BOS, 1.0, self.config.max_target_length)]

        for _ in range(self.config.max_target_length):
            results = self.infer_iteration(top_vec, gui_vec, x_input, results)

        return results[0].text()

    def infer_iteration(self, top_vec: torch.Tensor, gui_vec: torch.Tensor, x_input: dict, results: List['BeamSearchResult']):
        results_next: List[BeamSearchResult] = []

        for result in results:
            if result.is_eos():
                results_next.append(result)
            else:
                target = result.token_ids.reshape(1, self.config.max_target_length)
                state = self.dec.create_decoder_state(x_input['token_ids'], x_input['token_ids'])
                dec_out, state = self.dec(target, top_vec, gui_vec, state)
                output = self.gen(dec_out)

                top_k = torch.topk(output[0, result.length() - 1], k=self.config.beam_k)
                probs = torch.exp(top_k.values)
                token_ids = top_k.indices
                tokens = self.data_module.tokenizer.convert_ids_to_tokens(token_ids)

                for i in range(self.config.beam_k):
                    results_next.append(result.append(token_ids[i], tokens[i], probs[i]))

        results_next = sorted(results_next, key=lambda r: r.beam_prob(self.config.beam_alpha, self.config.beam_m), reverse=True)
        results_next = list(filter(lambda r: not r.is_eos() or r.length() > 3, results_next))
        results_next = results_next[:self.config.beam_k]
        return results_next

    def predict(self, x_input: dict, target: torch.Tensor):
        """
        TODO: Document
        """
        self.enc.to(self.device)
        self.dec.to(self.device)
        self.gen.to(self.device)

        top_vec, gui_vec = self.enc(x_input)
        state = self.dec.create_decoder_state(x_input['token_ids'], x_input['token_ids'])  # TODO: Is this actually needed in state? I think not... Replace 2nd parameter with guidance signal.
        dec_out, state = self.dec(target, top_vec, gui_vec, state)
        output = self.gen(dec_out)

        return output, state

    def shared_step(self, step, batch, batch_idx):
        x_input = batch['x_input']
        target = batch['y']['token_ids']
        dec_out, state = self.predict(x_input, target)

        target_shifted = target.roll(-1, 1)
        target_shifted[:, target_shifted.shape[1] - 1] = 0

        loss = self.loss_func(dec_out.view(-1, self.enc.bert.config.vocab_size), target_shifted.view(-1))

        if torch.isnan(loss):
            raise RuntimeError(f'Loss function returned NaN result in step {step} for batch {batch_idx}')

        self.log(f'{step}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self.shared_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step('test', batch, batch_idx)

    def configure_optimizers(self):
        enc_params = [param for name, param in self.named_parameters() if name.startswith('enc.bert')]
        enc_optim = torch.optim.Adam(enc_params, self.config.encoder_optim_lr, self.config.encoder_optim_beta, self.config.encoder_optim_eps)

        dec_params = [param for name, param in self.named_parameters() if not name.startswith('enc.bert')]
        dec_optim = torch.optim.Adam(dec_params, self.config.decoder_optim_lr, self.config.decoder_optim_beta, self.config.decoder_optim_eps)

        return [enc_optim, dec_optim]

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: torch.optim.Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None) -> None:

        if optimizer_idx == 0:
            lr = self.config.encoder_optim_lr
            warmup = self.config.encoder_optim_warmup_steps
        else:
            lr = self.config.decoder_optim_lr
            warmup = self.config.decoder_optim_warmup_steps

        lr = lr * min((self.trainer.global_step + 1) ** (-.5), (self.trainer.global_step + 1) * warmup ** (-1.5))

        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step(closure=optimizer_closure)


class GuidedExtSum(pl.LightningModule):
    """
    Bert based extractive summarizer.
    """

    def __init__(self, cfg: GuidedSummarizationConfig, data_module: GuidedSummarizationDataModule):
        super(GuidedExtSum, self).__init__()
        self.config = cfg
        self.data_module = data_module

        self.bert = Bert(cfg)
        self.enc = ExtSumTransformerEncoder(self.bert.model.config.hidden_size)

    def forward(self, x: List[str]):
        prepared_data = self.data_module.preprocess(x)
        results = []

        # TODO: correct implementation

        for batch in iter(prepared_data):
            scores = self.predict(batch['x_input'])
            results += [scores]

        return results

    def predict(self, x_input: dict):
        """
        Makes a multi-label prediction (extractive summarization - which sentences should be selected).

        The input dictionary is expected to contain:
        - `token_ids` - Tokenized input sequences (BERT tokenizer).
        - `attention_mask` - Attention mask as created by BERT tokenizer.
        - `token_type_ids` - Segment ids to mark different sentences.
        - `cls_indices` - A tensor containing indices of sentence classifiers within `token_ids`
        - `cls_mask` - Attention mask for sentences used in the document embedding.

        The dimensions if the inputs are:
        - `token_ids` - [batch_size, ..., ...]

        :param x_input: The input dictionary.

        Returns:
            The predicted possibilities for each sentence to be selected for the summary.
        """
        self.bert.to(self.device)
        self.enc.to(self.device)

        top_vec = self.bert(x_input)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), x_input['cls_indices'].type(torch.LongTensor)]
        return self.enc(sents_vec, x_input['cls_mask'])

    def shared_step(self, step, batch):
        x_input = batch['x_input']
        y = batch['y']
        z = self.predict(x_input)

        loss = F.binary_cross_entropy(z, y['sentence_mask'].float())
        self.log(f'{step}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step('train', batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_step('val', batch)

    def test_step(self, batch, batch_idx):
        return self.shared_step('test', batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00002)


#
# Torch Modules
#
class ExtSumTransformerEncoder(nn.Module):
    """
    An transformer encoder layer which takes the sentence tokens from BERT and
    encodes the whole document (based on sentences). Additionally this module
    adds layers to classify, whether sentence should be selected for extractive
    summary.
    """

    def __init__(self, d_model: int, layers: int = 2, heads: int = 8, dim_ff: int = 2048, dropout: float = 0.2):
        super(ExtSumTransformerEncoder, self).__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers, nn.LayerNorm(d_model, eps=1e-6))
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vec, cls_mask):
        x = self.pos_enc(top_vec)
        x = self.encoder(x, src_key_padding_mask=cls_mask.bool())
        x = self.layer_norm(x)
        x = self.wo(x)
        x = self.sigmoid(x)
        x = x.squeeze(-1) * (1 - cls_mask)
        return x


class Bert(nn.Module):
    """
    A wrapper for a simple Bert encoder.
    """

    def __init__(self, cfg: GuidedSummarizationConfig):
        super(Bert, self).__init__()
        self.config = cfg
        self.model = AutoModel.from_pretrained(cfg.base_model_name)

    def forward(self, x: dict):
        outputs = self.model(x['token_ids'], attention_mask=x['attention_mask'], token_type_ids=x['segment_ids'])
        return outputs.last_hidden_state


class AbsSumTransformerEncoder(nn.Module):
    """
    A wrapper for the Bert-based based encoder part of the Guided Summarization model.
    """

    def __init__(self, config: GuidedSummarizationConfig):
        super(AbsSumTransformerEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(config.base_model_name)

        if config.max_input_length > 512:
            """
            If maximum sequence length is longer than BERT's default max sequence length,
            then we need to extend the positional embeddings layer of the model.
            """
            my_pos_embeddings = nn.Embedding(config.max_input_length, self.bert.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][None, :].repeat(config.max_input_length - 512, 1)
            self.bert.embeddings.position_embeddings = my_pos_embeddings

        self.input_transformer_encoder = nn.TransformerEncoderLayer(
            self.bert.config.hidden_size, config.encoder_heads, config.encoder_ff_dim, config.encoder_dropout, batch_first=True)

        self.guidance_transformer_encoder = nn.TransformerEncoderLayer(
            self.bert.config.hidden_size, config.encoder_heads, config.encoder_ff_dim, config.encoder_dropout, batch_first=True)

    def forward(self, x_input: dict):
        """
        Encodes the input provided by `x_input`. `x_input` is a dict containing `token_ids` and `attention_mask` as produced by
        Bert Tokenizer. Additionally `segment_ids` should be provided to mark different sentences, as described by GSum Paper.

        `segment_ids` is a tensor of shape [BATCH_SIZE x INPUT_SEQUENCE_LENGTH]

        :param x_input: The tokenized input sequence.

        Returns
            Encoded input sequence; Tensor of shape [BATCH_SIZE x INPUT_SEQUENCE_LENGTH x BERT_HIDDEN_SIZE]
        """
        input_vec = self.bert(x_input['token_ids'], attention_mask=x_input['attention_mask'], token_type_ids=x_input['segment_ids'])
        input_vec = self.input_transformer_encoder(input_vec['last_hidden_state'], src_key_padding_mask=(1 - x_input['attention_mask']).bool())
        """
        TODO: Implement guidance signal
        guidance_vec = self.bert(**x_guidance)
        guidance_vec = self.guidance_transformer_encoder(guidance_vec, 1 - x_guidance['attention_mask'])
        """

        return input_vec, input_vec


class AbsSumTransformerDecoderLayer(nn.Module):
    """
    Represents a single decoder layer for guided summarization.
    """

    @staticmethod
    def create_attention_mask(size):
        """
        Creates an attention mask based on expected target size.

        :param size: The target sequence length.

        Returns:
            A LongTensor of shape [1 x size x size]
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask

    def __init__(self, config: GuidedSummarizationConfig, d_model: int, heads: int, d_ff: int, dropout: float):
        """
        :param d_model: The dimensions of the model TODO
        :param heads: Number of heads for attention layers.
        :param d_ff: Dimensions of the inner FF layer.
        :param dropout: Dropout value for dropout layers.
        """
        super(AbsSumTransformerDecoderLayer, self).__init__()

        self.num_heads = heads

        self.self_attention = nn.MultiheadAttention(d_model, heads, dropout, batch_first=True)
        self.encoder_input_attention = nn.MultiheadAttention(d_model, heads, dropout, batch_first=True)
        self.encoder_signal_attention = nn.MultiheadAttention(d_model, heads, dropout, batch_first=True)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.input_normalization = nn.LayerNorm(d_model, eps=1e-6)
        self.query_normalization = nn.LayerNorm(d_model, eps=1e-6)
        self.signal_normalization = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', AbsSumTransformerDecoderLayer.create_attention_mask(config.max_target_length))

    def forward(self, target_embedded: torch.Tensor, encoder_input_context: torch.Tensor,
                encoder_signal_context: torch.Tensor, target_mask: torch.Tensor, encoder_input_mask: torch.Tensor,
                encoder_signal_mask: torch.Tensor, previous_target_embedded: torch.Tensor = None):
        """

        TODO document ...
        layer_cache not used from orig.
        step not used from orig ...
        """

        #
        # Self Attention
        #
        self_attention_mask = torch.gt(target_mask + self.mask, 0)  # TODO: Shouldn't I ignore self.mask for inference time?
        target_normalized = self.input_normalization(target_embedded)

        self_attention_mask = self_attention_mask.repeat(self.num_heads, 1, 1)
        encoder_signal_mask = encoder_signal_mask.repeat(self.num_heads, 1, 1)
        encoder_input_mask = encoder_input_mask.repeat(self.num_heads, 1, 1)

        if previous_target_embedded is not None:
            target = torch.cat([previous_target_embedded, target_normalized])
            self_attention_mask = None
        else:
            target = target_normalized

        # TODO: Expand Masks to Head count

        self_attn_out, _ = self.self_attention(target, target, target_normalized, attn_mask=self_attention_mask)  # TODO: Check Mask
        self_attn_out = self.drop(self_attn_out) + target_embedded

        #
        # Signal Context Cross Attention
        #
        signal_attn_query_normalized = self.query_normalization(self_attn_out)
        signal_attn_out, _ = self.encoder_signal_attention(encoder_signal_context,
                                                           encoder_signal_context,
                                                           signal_attn_query_normalized,
                                                           attn_mask=encoder_signal_mask)
        signal_attn_out = self.drop(signal_attn_out) + self_attn_out

        #
        # Input Context Cross Attention
        #
        input_attn_query_normalized = self.query_normalization(signal_attn_out)
        input_attn_out, _ = self.encoder_input_attention(encoder_input_context,
                                                         encoder_input_context,
                                                         input_attn_query_normalized,
                                                         attn_mask=encoder_input_mask)
        input_attn_out = self.drop(input_attn_out) + signal_attn_out

        return input_attn_out, target


class AbsSumTransformerDecoder(nn.Module):

    def __init__(self, config: GuidedSummarizationConfig, num_layers: int, d_model: int, heads: int, d_ff: int,
                 dropout: float, embedding: nn.Embedding, vocab_size: int):
        """
        The decoder building block for guided summarization.


        :param num_layers: The number of encoder layers.
        :param d_model: The dimensions of the model TODO
        :param heads: Number of heads for attention layers.
        :param d_ff: Dimensions of the inner FF layer.
        :param dropout: Dropout value for dropout layers.
        :param embedding: Embedding with copied weights from BERT's embedding layer; Should be equal to embedding layer of encoder.
        :param vocab_size: The size of the tokenizers vocabulary.
        """
        super(AbsSumTransformerDecoder, self).__init__()

        self.config = config
        self.num_layers = num_layers
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(self.embedding.embedding_dim, dropout)
        self.vocab_size = vocab_size

        self.transformer_layers = nn.ModuleList([AbsSumTransformerDecoderLayer(config, d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.ff = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, target: torch.Tensor, encoder_input_context: torch.Tensor, encoder_signal_context: torch.Tensor,
                state: 'AbsSumTransformerDecoderState'):
        """
        TODO

        Args:
            target: FloatTensor of shape [batch_size x target_seq_length] containing token-ids of the target sequence.
            encoder_input_context: FloatTensor of shape [batch_size x document_vector_length x encoder_dimensions] containing encoded input document.
            encoder_signal_context:
            state:

        Returns:

        """
        batch_size = target.shape[0]

        #
        # Prepare padding masks for input, signals and target
        #
        target_embedded = self.embedding(target)
        target_embedded = self.positional_encoding(target_embedded)

        padding_idx = self.embedding.padding_idx
        target_mask = target.data.eq(padding_idx).unsqueeze(1).expand(
            batch_size, self.config.max_target_length, self.config.max_target_length)

        encoder_input_mask = state.encoder_input_source.data.eq(padding_idx).unsqueeze(1).expand(
            batch_size, self.config.max_target_length, self.config.max_input_length)

        encoder_signal_mask = state.encoder_input_signal.data.eq(padding_idx).unsqueeze(1).expand(
            batch_size, self.config.max_target_length, self.config.max_input_signal_length)

        #
        # Run forward
        #
        saved_inputs = []
        output = target_embedded

        for i in range(self.num_layers):
            if state.previous_input is not None:
                prev_lay_input = state.previous_layer_inputs[i]
            else:
                prev_lay_input = None

            output, all_input = self.transformer_layers[i](
                output, encoder_input_context, encoder_signal_context,
                target_mask, encoder_input_mask, encoder_signal_mask, prev_lay_input)

            saved_inputs.append(all_input)

        output = self.ff(output)

        #
        # Prepare result and retun
        #
        saved_inputs = torch.stack(saved_inputs)
        state = state.update_state(target, saved_inputs)

        return output, state

    def create_decoder_state(self, encoder_input_source, encoder_input_signal) -> 'AbsSumTransformerDecoderState':
        """
        TODO: Purpose/ Description?
        TODO: As of now we do not use caching as in GSum impl (also not used there, just implemented)
        """
        return AbsSumTransformerDecoderState(encoder_input_source, encoder_input_signal)


class AbsSumTransformerDecoderState(object):
    """
    Helper class to support decoding.
    """

    def __init__(self, encoder_input_source, encoder_input_signal):
        """
        TODO descriptions and dimensions.
        Args:
            encoder_input_source:
            encoder_input_signal:
        """
        self.encoder_input_source = encoder_input_source
        self.encoder_input_signal = encoder_input_signal

        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    def update_state(self, new_input, previous_layer_inputs) -> 'AbsSumTransformerDecoderState':
        """
        TODO: Why not just setting values?
        Args:
            new_input:
            previous_layer_inputs:

        Returns:

        """
        state = AbsSumTransformerDecoderState(self.encoder_input_source, self.encoder_input_signal)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that the two can be summed.
    Here, we use sine and cosine functions of different frequencies.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = PositionwiseFeedForward.gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        inter = self.dropout_1(PositionwiseFeedForward.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and: p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class BeamSearchResult:

    def __init__(self, token_ids: torch.Tensor, tokens: List[str], probs: List[float]):
        self.token_ids = token_ids
        self.tokens = tokens
        self.probs = probs

    @staticmethod
    def create(token_id: int, token: str, prob: float, input_length: int) -> 'BeamSearchResult':
        token_ids = torch.zeros(input_length).type(torch.IntTensor)
        token_ids[0] = token_id
        return BeamSearchResult(token_ids, [token], [prob])

    def prob(self, m: int) -> float:
        return functools.reduce(lambda p1, p2: p1 * p2, self.probs[-m:])

    def beam_prob(self, alpha: float, m: int) -> float:
        return np.log(self.prob(m)) * (1/(self.length() ** alpha))

    def text(self) -> str:
        # TODO filter/ transform
        return ' '.join(self.tokens)

    def append(self, token_id: int, token: str, prob: float) -> 'BeamSearchResult':
        token_ids = self.token_ids
        token_ids[len(token)] = token_id

        return BeamSearchResult(token_ids, self.tokens + [token], self.probs + [prob])

    def is_eos(self) -> bool:
        return self.tokens[self.length() - 1] == TARGET_EOS

    def length(self) -> int:
        return len(self.tokens)

    def __str__(self):
        return f'BeamSearchResult(`{self.text()})'
