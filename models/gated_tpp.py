import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self,
                 num_types, d_model, t_max=200):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.type_emb = nn.Embedding(num_types + 1, d_model, padding_idx=0)
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)])
        self.kernel = squared_exponential_kernel
        self.t_max = 200

    def forward(self, event_type, event_time):
        """ Encode event sequences via kernel functions """

        ## Temporal Encoding
        temp_enc = event_time.unsqueeze(-1) / self.position_vec
        temp_enc[:, :, 0::2] = torch.sin(temp_enc[:, :, 0::2])
        temp_enc[:, :, 1::2] = torch.cos(temp_enc[:, :, 1::2])

        ## Type Encoding
        type_embedding = self.type_emb(event_type) * math.sqrt(
            self.d_model)  ## Scale the embedding with the hidden vector size

        embedding = torch.cat([temp_enc, type_embedding], dim=-1)

        ## Future Masking
        subsequent_mask = get_subsequent_mask(event_type)

        ## Time Scores
        normalized_event_time = event_time / self.t_max
        xt_bar = normalized_event_time.unsqueeze(1). \
            expand(normalized_event_time.size(0), normalized_event_time.size(1), normalized_event_time.size(1))
        xt = xt_bar.transpose(1, 2)
        scores = self.kernel((xt, xt_bar))
        scores = (scores * subsequent_mask).masked_fill_(subsequent_mask == 0, value=-1000)
        scores = F.softmax(scores, dim=-1)

        return scores, embedding


class Decoder(nn.Module):
    """ A non parametric decoder. """

    def __init__(self,
                 num_types, d_model):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.GAN =GenerativeAdversarialNetwork()


class GenerativeAdversarialNetwork(nn.Module):

    def __init__(self,
                 num_types, d_model):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types




class gated_TPP(nn.Module):

    def __init__(self,
                 num_types, d_model, t_max=200):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.encoder = Encoder(num_types, d_model, t_max=t_max)

    def forward(self, event_type, event_time):
        scores, embeddings = self.encoder(event_type, event_time)
        history_vector = torch.matmul(scores, embeddings)


        return history_vector


def squared_exponential_kernel(x, sigma=10, lambd=0.005, norm=2):
    d = torch.abs(x[0] - x[1]) ** norm

    return (sigma ** 2) * torch.exp(-(d ** 2) / lambd ** 2)


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    subsequent_mask = (subsequent_mask - 1) ** 2
    return subsequent_mask
