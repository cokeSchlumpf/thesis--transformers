import copy
import functools
import math
import numpy as np
import pytorch_lightning as pl
import re
import time
import torch
import torch.nn.functional as F

from torch import nn
from transformers import AutoModel
from typing import Callable, List, Optional, Tuple

from lib.text_preprocessing import clean_html, preprocess_text, simple_punctuation_only, to_lower
from lib.utils import extract_sentence_tokens

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .preprocess_inputs import TARGET_BOS, TARGET_EOS, TARGET_SEP


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
        self.enc = AbsSumTransformerEncoder(cfg, data_module)  # AbsTransformerEncoder

        # Shared embedding layer
        embedding = nn.Embedding(self.enc.bert.model.config.vocab_size, self.enc.bert.model.config.hidden_size, padding_idx=0)
        embedding.weight = copy.deepcopy(self.enc.bert.model.embeddings.word_embeddings.weight)

        # Decoder
        self.dec = AbsSumTransformerDecoder(
            cfg, cfg.decoder_layers, cfg.decoder_dim, cfg.decoder_heads, cfg.decoder_ff_dim, cfg.decoder_dropout,
            embedding, self.enc.bert.model.config.vocab_size)

        # Generator
        self.gen = nn.Sequential(
            nn.Linear(self.enc.bert.model.config.hidden_size, self.enc.bert.model.config.vocab_size),
            nn.LogSoftmax(dim=-1))

        self.loss_func = LabelSmoothingLoss(cfg.label_smoothing, self.enc.bert.model.config.vocab_size, 0)

        #
        # TODO: Weights initialization?
        #

    def forward(self, x: List[str]) -> Tuple[List['SummarizationResult'], float]:
        print(self.device)

        self.enc.to(self.device)
        self.dec.to(self.device)
        self.gen.to(self.device)
        start = time.time()

        #
        # prepare batch
        #
        x_prepared = list([p for p in self.data_module.preprocess(x)])
        token_ids = torch.cat([p['x_input']['token_ids'] for p in x_prepared]).to(self.device)
        attention_masks = torch.cat([p['x_input']['attention_mask'] for p in x_prepared]).to(self.device)
        segment_ids = torch.cat([p['x_input']['segment_ids'] for p in x_prepared]).to(self.device)

        x_input_batch = {
            'token_ids': token_ids,
            'attention_mask': attention_masks,
            'segment_ids': segment_ids
        }

        token_ids = torch.cat([p['x_guidance']['token_ids'] for p in x_prepared]).to(self.device)
        attention_masks = torch.cat([p['x_guidance']['attention_mask'] for p in x_prepared]).to(self.device)
        segment_ids = torch.cat([p['x_guidance']['segment_ids'] for p in x_prepared]).to(self.device)

        x_guidance_batch = {
            'token_ids': token_ids,
            'attention_mask': attention_masks,
            'segment_ids': segment_ids
        }

        #
        # Encoder
        #
        top_vec, gui_vec = self.enc(x_input_batch, x_guidance_batch)

        #
        # Decoder
        #
        beam_search: List['BeamSearchState'] = []
        for i in range(top_vec.shape[0]):
            beam_search.append(
                BeamSearchState(x[i], top_vec[i], gui_vec[i], self.data_module, self.config, self.device))

        while True:
            # Collect inputs for next decoder batch and store which entry maps to which search state
            results_in_progress: List[BeamSearchResult] = []
            results_search_map: List[int] = []
            batch_x_input_token_ids: List[torch.Tensor] = []
            batch_x_guidance_token_ids: List[torch.Tensor] = []
            batch_topic_vecs: List[torch.Tensor] = []
            batch_gui_vecs: List[torch.Tensor] = []

            for i, s in enumerate(beam_search):
                # Add open results to batch until max. batch size is used
                results_in_progress_from_search = s.results_in_progress()
                results_in_progress = results_in_progress + results_in_progress_from_search
                results_search_map = results_search_map + ([i] * len(results_in_progress_from_search))
                batch_x_input_token_ids = batch_x_input_token_ids + (
                            [x_input_batch['token_ids'][i]] * len(results_in_progress_from_search))
                batch_x_guidance_token_ids = batch_x_guidance_token_ids + (
                            [x_guidance_batch['token_ids'][i]] * len(results_in_progress_from_search))
                batch_topic_vecs = batch_topic_vecs + ([s.top_vec] * len(results_in_progress_from_search))
                batch_gui_vecs = batch_gui_vecs + ([s.gui_vec] * len(results_in_progress_from_search))

                if len(results_in_progress) > self.config.batch_sizes[2]:
                    break

            # stop search if no open search results are found.
            if len(results_in_progress) == 0:
                print()
                break
            else:
                print('.', end='')

            # Remove potential batch overflow from batch
            results_in_progress = results_in_progress[:self.config.batch_sizes[2]]
            results_search_map = results_search_map[:self.config.batch_sizes[2]]
            batch_x_input_token_ids = batch_x_input_token_ids[:self.config.batch_sizes[2]]
            batch_x_guidance_token_ids = batch_x_guidance_token_ids[:self.config.batch_sizes[2]]
            batch_topic_vecs = batch_topic_vecs[:self.config.batch_sizes[2]]
            batch_gui_vecs = batch_gui_vecs[:self.config.batch_sizes[2]]

            # Prepare decoder input
            target = torch.cat(
                [result.token_ids.reshape(1, self.config.max_target_length) for result in results_in_progress])
            state = self.dec.create_decoder_state(torch.stack(batch_x_input_token_ids),
                                                  torch.stack(batch_x_guidance_token_ids))

            dec_out, state = self.dec(target, torch.stack(batch_topic_vecs), torch.stack(batch_gui_vecs), state)
            output = self.gen(dec_out)

            # Update search results
            for i, r in enumerate(results_in_progress):
                search_state: 'BeamSearchState' = beam_search[results_search_map[i]]
                top_k = torch.topk(output[i, r.length() - 1], k=self.config.beam_k)
                probs = torch.exp(top_k.values)
                token_ids = top_k.indices
                tokens: List[str] = self.data_module.tokenizer.convert_ids_to_tokens(token_ids)

                results_next = [r.append(token_ids[j], t, probs[j]) for j, t in enumerate(tokens)]
                search_state.replace_result(r, results_next)

            # Clean and sort results
            for s in beam_search:
                s.sort_and_clean()

        end = time.time()
        return [s.results[0] for s in beam_search], end - start

    def predict(self, x_input: dict, x_guidance: dict, target: torch.Tensor):
        """
        TODO: Document
        """
        self.enc.to(self.device)
        self.dec.to(self.device)
        self.gen.to(self.device)

        top_vec, gui_vec = self.enc(x_input, x_guidance)
        state = self.dec.create_decoder_state(x_input['token_ids'], x_guidance['token_ids'])
        dec_out, state = self.dec(target, top_vec, gui_vec, state)
        output = self.gen(dec_out)

        return output, state

    def shared_step(self, step, batch, batch_idx):
        x_input = batch['x_input']
        x_guidance = batch['x_guidance']
        target = batch['y']['token_ids']
        dec_out, state = self.predict(x_input, x_guidance, target)

        target_shifted = target.roll(-1, 1)
        target_shifted[:, target_shifted.shape[1] - 1] = 0

        loss = self.loss_func(dec_out.view(-1, self.enc.bert.model.config.vocab_size), target_shifted.view(-1))

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
        enc_params = [param for name, param in self.named_parameters() if name.startswith('enc.bert.model')]
        enc_optim = torch.optim.Adam(enc_params, self.config.encoder_optim_lr, self.config.encoder_optim_beta,
                                     self.config.encoder_optim_eps)

        dec_params = [param for name, param in self.named_parameters() if not name.startswith('enc.bert.model')]
        dec_optim = torch.optim.Adam(dec_params, self.config.decoder_optim_lr, self.config.decoder_optim_beta,
                                     self.config.decoder_optim_eps)

        return [dec_optim, enc_optim]

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

        if optimizer_idx == 1:
            lr = self.config.encoder_optim_lr
            warmup = self.config.encoder_optim_warmup_steps
        else:
            lr = self.config.decoder_optim_lr
            warmup = self.config.decoder_optim_warmup_steps

        lr = lr * min((self.trainer.global_step + 1) ** (-.5), (self.trainer.global_step + 1) * warmup ** (-1.5))

        self.log(f'opt_{optimizer_idx}_lr', lr, prog_bar=True)

        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step(closure=optimizer_closure)

    def toggle_optimizer(self, optimizer: torch.optim.Optimizer, optimizer_idx: int):
        """
        Makes sure only the gradients of the current optimizer's parameters are calculated
        in the training step to prevent dangling gradients in multiple-optimizer setup.

        .. note:: Only called when using multiple optimizers

        Args:
            optimizer: Current optimizer used in training_loop
            optimizer_idx: Current optimizer idx in training_loop
        """

        # Iterate over all optimizer parameters to preserve their `requires_grad` information
        # in case these are pre-defined during `configure_optimizers`
        param_requires_grad_state = self._param_requires_grad_state
        for opt in self.optimizers(use_pl_optimizer=False):
            for group in opt.param_groups:
                for param in group['params']:
                    # If a param already appear in param_requires_grad_state, continue
                    if param not in param_requires_grad_state:
                        param_requires_grad_state[param] = param.requires_grad

                    param.requires_grad = False

        # Then iterate over the current optimizer's parameters and set its `requires_grad`
        # properties accordingly
        for group in optimizer.param_groups:
            for param in group['params']:
                param.requires_grad = param_requires_grad_state[param]
        self._param_requires_grad_state = param_requires_grad_state

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)


class GuidedExtSum(pl.LightningModule):
    """
    Bert based extractive summarizer.
    """

    def __init__(self, cfg: GuidedSummarizationConfig, data_module: GuidedSummarizationDataModule):
        super(GuidedExtSum, self).__init__()
        self.config = cfg
        self.data_module = data_module

        self.bert = Bert(cfg)
        self.enc = ExtSumTransformerEncoder(self.bert.model.config.hidden_size, dropout=self.config.encoder_dropout)
        self.loss = nn.BCELoss(reduction='none')

        for p in self.enc.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: List[str], threshold: float = 0.7) -> Tuple[List['SummarizationResult'], float]:
        start = time.time()

        #
        # prepare batch
        #
        x_prepared = list([p for p in self.data_module.preprocess(x)])
        token_ids = torch.cat([p['x_input']['token_ids'] for p in x_prepared]).to(self.device)
        attention_mask = torch.cat([p['x_input']['attention_mask'] for p in x_prepared]).to(self.device)
        segment_ids = torch.cat([p['x_input']['segment_ids'] for p in x_prepared]).to(self.device)
        cls_indices = torch.cat([p['x_input']['cls_indices'] for p in x_prepared]).to(self.device)
        cls_mask = torch.cat([p['x_input']['cls_mask'] for p in x_prepared]).to(self.device)

        x_input_batch = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'segment_ids': segment_ids,
            'cls_indices': cls_indices,
            'cls_mask': cls_mask
        }

        scores = self.predict(x_input_batch)
        pipeline = [simple_punctuation_only, to_lower]
        result = []

        for i in range(scores.shape[0]):
            sample = x[i]
            cleaned = preprocess_text(sample, self.data_module.lang, pipeline, [clean_html])
            sentences = extract_sentence_tokens(self.data_module.lang, cleaned)
            sentences = filter(lambda s: len(s) >= self.config.min_sentence_tokens, sentences)
            sentences = map(lambda s: ' '.join(s), sentences)
            sentences = list(sentences)

            sentences_selected = []
            for j in torch.topk(scores[i], 3).indices:
                if len(sentences) > j:
                    sentences_selected.append(sentences[j])

            sentences = ' '.join(sentences_selected)
            sentences = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', sentences)
            result.append(SimpleSummarizationResult(sentences))

        end = time.time()
        return result, end - start

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
        sents_vec = sents_vec * (1 - x_input['cls_mask'][:, :, None]).float()
        sents_vec = self.enc(sents_vec, x_input['cls_mask'])
        sents_vec = sents_vec.squeeze(-1)
        return sents_vec

    def shared_step(self, step, batch):
        x_input = batch['x_input']
        y = batch['y']
        z = self.predict(x_input)

        loss = self.loss(z, y['sentence_mask'].float() * (1 - x_input['cls_mask']).float())
        loss = (loss * (1 - x_input['cls_mask']).float()).sum() / loss.numel()
        self.log(f'{step}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step('train', batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_step('val', batch)

    def test_step(self, batch, batch_idx):
        return self.shared_step('test', batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.config.encoder_optim_lr, betas = self.config.encoder_optim_beta, eps=self.config.encoder_optim_eps)

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

        warmup = self.config.encoder_optim_warmup_steps
        lr = 2 * np.exp(-3) * min((self.trainer.global_step + 1) ** (-.5), (self.trainer.global_step + 1) * warmup ** (-1.5))

        self.log(f'opt_{optimizer_idx}_lr', lr, prog_bar=True)

        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step(closure=optimizer_closure)


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
        self.pos_enc = PositionalEncoding2(d_model, dropout)

        # encoder_layer = nn.TransformerEncoderLayer(d_model, heads, dim_ff, dropout, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.layers = layers
        self.encoder_layers = nn.ModuleList([ExtTransformerEncoderLayer(d_model, heads, dim_ff, dropout) for _ in range(layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vec, cls_mask):
        max_sentences = top_vec.shape[1]
        pos_emb = self.pos_enc.pe[:, :max_sentences]
        x = top_vec * (1 - cls_mask[:, :, None]).float()
        x = x + pos_emb

        for i in range(self.layers):
            x = self.encoder_layers[i](i, x, cls_mask)

        x = self.layer_norm(x)
        x = self.wo(x)
        x = self.sigmoid(x)
        x = x.squeeze(-1) * (1 - cls_mask).float()
        return x


class Bert(nn.Module):
    """
    A wrapper for a simple Bert encoder.
    """

    def __init__(self, cfg: GuidedSummarizationConfig):
        super(Bert, self).__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.base_model_name)

    def forward(self, x: dict):
        if self.cfg.base_model_name.startswith('bert-') or ('electra' in self.cfg.base_model_name):
            outputs = self.model(x['token_ids'], attention_mask=x['attention_mask'], token_type_ids=x['segment_ids'])
        elif self.cfg.base_model_name.startswith('distilbert-'):
            outputs = self.model(x['token_ids'])
        else:
            raise Exception(f"unknown model type `{self.cfg.base_model_name}`")

        return outputs.last_hidden_state


class AbsSumTransformerEncoder(nn.Module):
    """
    A wrapper for the Bert-based based encoder part of the Guided Summarization model.
    """

    def __init__(self, config: GuidedSummarizationConfig, data_module: GuidedSummarizationDataModule):
        super(AbsSumTransformerEncoder, self).__init__()

        if config.base_model_pretrained is None:
            self.bert = Bert(config)

            if config.max_input_length > 512:
                """
                If maximum sequence length is longer than BERT's default max sequence length,
                then we need to extend the positional embeddings layer of the model.
                """
                my_pos_embeddings = nn.Embedding(config.max_input_length, self.bert.config.hidden_size)
                my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
                my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][
                                                      None, :].repeat(config.max_input_length - 512, 1)
                self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        else:
            ext = GuidedExtSum.load_from_checkpoint(config.base_model_pretrained, cfg=config, data_module=data_module)
            self.bert = ext.bert

        self.input_transformer_encoder = nn.TransformerEncoderLayer(
            self.bert.model.config.hidden_size, config.encoder_heads, config.encoder_ff_dim, config.encoder_dropout,
            batch_first=True)

        self.guidance_transformer_encoder = nn.TransformerEncoderLayer(
            self.bert.model.config.hidden_size, config.encoder_heads, config.encoder_ff_dim, config.encoder_dropout,
            batch_first=True)

    def forward(self, x_input: dict, x_guidance: dict):
        """
        Encodes the input provided by `x_input`. `x_input` is a dict containing `token_ids` and `attention_mask` as produced by
        Bert Tokenizer. Additionally `segment_ids` should be provided to mark different sentences, as described by GSum Paper.

        `segment_ids` is a tensor of shape [BATCH_SIZE x INPUT_SEQUENCE_LENGTH]

        :param x_input: The tokenized input sequence.
        :param x_guidance: The encoded guidance signal.

        Returns
            Encoded input sequence; Tensor of shape [BATCH_SIZE x INPUT_SEQUENCE_LENGTH x BERT_HIDDEN_SIZE]
        """
        input_vec = self.bert(x_input)
        input_vec = self.input_transformer_encoder(input_vec, src_key_padding_mask=(1 - x_input['attention_mask']).bool())

        guidance_vec = self.bert(x_guidance)
        guidance_vec = self.guidance_transformer_encoder(guidance_vec, src_key_padding_mask=(1 - x_guidance['attention_mask']).bool())

        return input_vec, guidance_vec


class ExtTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super(ExtTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iteration, inputs, mask):
        if iteration != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, (1 - mask).bool())
        output = self.dropout(context) + inputs
        output = self.feed_forward(output)
        return output


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
        self_attention_mask = torch.gt(target_mask + self.mask, 0)
        target_normalized = self.input_normalization(target_embedded)

        self_attention_mask = torch.repeat_interleave(self_attention_mask, self.num_heads, dim=0)
        encoder_signal_mask = torch.repeat_interleave(encoder_signal_mask, self.num_heads, dim=0)
        encoder_input_mask = torch.repeat_interleave(encoder_input_mask, self.num_heads, dim=0)

        if previous_target_embedded is not None:
            target = torch.cat([previous_target_embedded, target_normalized])
            self_attention_mask = None
        else:
            target = target_normalized

        self_attn_out, _ = self.self_attention(target_normalized, target, target, attn_mask=self_attention_mask)
        self_attn_out = self.drop(self_attn_out) + target_embedded

        #
        # Signal Context Cross Attention
        #
        signal_attn_query_normalized = self.query_normalization(self_attn_out)
        signal_attn_out, _ = self.encoder_signal_attention(signal_attn_query_normalized,
                                                           encoder_signal_context,
                                                           encoder_signal_context,
                                                           attn_mask=encoder_signal_mask)
        signal_attn_out = self.drop(signal_attn_out) + self_attn_out

        #
        # Input Context Cross Attention
        #
        input_attn_query_normalized = self.query_normalization(signal_attn_out)
        input_attn_out, _ = self.encoder_input_attention(input_attn_query_normalized,
                                                         encoder_input_context,
                                                         encoder_input_context,
                                                         attn_mask=encoder_input_mask)
        input_attn_out = self.drop(input_attn_out) + signal_attn_out
        input_attn_out = self.feed_forward(input_attn_out)
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

        self.transformer_layers = nn.ModuleList(
            [AbsSumTransformerDecoderLayer(config, d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.ff = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, target: torch.Tensor, encoder_input_context: torch.Tensor, encoder_signal_context: torch.Tensor,
                state: 'AbsSumTransformerDecoderState'):
        """
        TODO

        Args:
            target: FloatTensor of shape [batch_size x target_seq_length] containing token-ids of the target sequence.
            encoder_input_context: FloatTensor of shape [batch_size x document_vector_length x encoder_dimensions] containing encoded input document.
            encoder_signal_context: Guidance signal of shape [batch_size x guidance_signal_length x encoder_dimensions]
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


class PositionalEncoding2(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


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


class MultiHeadedAttention(nn.Module):

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear

        if self.use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None, layer_cache=None, type=None, predefined_graph_1=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
            elif type == "z_context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["z_memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["z_memory_keys"], \
                                     layer_cache["z_memory_values"]
                    layer_cache["z_memory_keys"] = key
                    layer_cache["z_memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if not predefined_graph_1 is None:
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if self.use_final_linear:
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output, attn
        else:
            context = torch.matmul(drop_attn, value)
            return context, attn


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


class BeamSearchState:

    def __init__(self, input: str, top_vec: torch.Tensor, gui_vec: torch.Tensor,
                 data_module: GuidedSummarizationDataModule, config: GuidedSummarizationConfig, device: str):
        self.input = input
        self.top_vec = top_vec
        self.gui_vec = gui_vec
        self.results: List['BeamSearchResult'] = [
            BeamSearchResult.create(data_module.tokenizer.convert_tokens_to_ids(TARGET_BOS), TARGET_BOS, 1.0,
                                    config.max_target_length, device)]
        self.started = time.time()
        self.k = config.beam_k
        self.alpha = config.beam_alpha

    def results_in_progress(self) -> List['BeamSearchResult']:
        return [result for result in self.results if not result.is_eos()]

    def replace_result(self, existing: 'BeamSearchResult', next_results: List['BeamSearchResult']) -> None:
        self.results.remove(existing)
        self.results = self.results + next_results

    def sort_and_clean(self) -> None:
        self.results = sorted(self.results, key=lambda r: r.beam_prob(self.alpha), reverse=True)
        self.results = list(filter(lambda r: (not r.is_eos() or r.length() > 3) and not r.is_repeating() and not r.is_complete(), self.results))
        self.results = self.results[:self.k]


class SummarizationResult:

    def text(self) -> str:
        pass


class SimpleSummarizationResult(SummarizationResult):

    def __init__(self, text: str):
        self.txt = text

    def text(self) -> str:
        return self.txt


class BeamSearchResult(SummarizationResult):

    def __init__(self, token_ids: torch.Tensor, tokens: List[str], probs: List[float],
                 prob_cache: Optional[float] = None, max_length: int = 256):
        self.token_ids = token_ids
        self.tokens = tokens
        self.probs = probs

        self.prob_cache = prob_cache
        self.beam_prob_cache = None
        self.max_length = max_length

    @staticmethod
    def create(token_id: int, token: str, prob: float, input_length: int, device: str) -> 'BeamSearchResult':
        token_ids = torch.zeros(input_length).type(torch.IntTensor).to(device)
        token_ids[0] = token_id
        return BeamSearchResult(token_ids, [token], [prob])

    def prob(self) -> float:
        if self.prob_cache is None:
            self.prob_cache = functools.reduce(lambda p1, p2: p1 * p2, self.probs)

        return self.prob_cache

    def beam_prob(self, alpha: float) -> float:
        if self.beam_prob_cache is None:
            self.beam_prob_cache = (torch.tensor(self.prob()).log() * (1 / (self.length() ** alpha))).item()

        return self.beam_prob_cache

    def text(self) -> str:
        cleaned = filter(lambda t: t != TARGET_SEP and t != TARGET_BOS and t != TARGET_EOS, self.tokens)
        cleaned = ' '.join(cleaned)
        cleaned = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', cleaned)
        return cleaned

    def append(self, token_id: int, token: str, prob: float) -> 'BeamSearchResult':
        token_ids = self.token_ids.detach().clone()
        token_ids[len(self.tokens)] = token_id

        return BeamSearchResult(token_ids, self.tokens + [token], self.probs + [prob], self.prob() * prob)

    def is_eos(self) -> bool:
        return self.tokens[-1] == TARGET_EOS or len(self.tokens) >= self.max_length

    def is_repeating(self) -> bool:
        if self.length() <= 1:
            return False
        else:
            return (self.tokens[-1] == self.tokens[-2]) or \
                   (self.length() >= 4 and self.tokens[-1] == self.tokens[-3] and self.tokens[-2] == self.tokens[-4]) or \
                   (self.length() >= 6 and self.tokens[-1] == self.tokens[-4] and self.tokens[-2] == self.tokens[-5] and
                    self.tokens[-3] == self.tokens[-6])

    def is_complete(self, n: int = 2) -> bool:
        return self.tokens.count('.') >= n or (len(self.tokens) > 30 and self.tokens.count('[UNK]') > 0.2 * len(self.tokens))

    def length(self) -> int:
        return len(self.tokens)

    def __str__(self):
        return f'BeamSearchResult(`{self.text()}`)'
