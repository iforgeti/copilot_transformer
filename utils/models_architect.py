from queue import PriorityQueue
import operator
from torch import nn
import torch
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("model/code-search-net-tokenizer-mod")

SOS_IDX = tokenizer.bos_token_id
EOS_IDX = tokenizer.eos_token_id
PAD_IDX = tokenizer.pad_token_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads  #make sure it's divisible....
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc   = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, q, k, v, mask = None):
        b = q.shape[0]
        
        Q = self.fc_q(q)
        K = self.fc_k(k)
        V = self.fc_v(v)
        #Q, K, V = [b, l, h]
        
        #reshape them into head_dim
        #reshape them to [b, n_heads, l, head_dim]
        Q = Q.view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #Q, K, V = [b, n_heads, l, head_dim]
        
        #e = QK/sqrt(dk)
        e = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #e: [b, n_heads, ql, kl]
        
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e10)
            
        a = torch.softmax(e, dim=-1)
        
        #eV
        x = torch.matmul(self.dropout(a), V)
        #x: [b, n_heads, ql, head_dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        #x: [b, ql, n_heads, head_dim]
        
        #concat them together
        x = x.view(b, -1, self.hid_dim)
        #x: [b, ql, h]
        
        x = self.fc(x)
        #x = [b, ql, h]
        
        return x, a

class PositionwiseFeedforwardLayer(nn.Module):
    
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))
    

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.norm_att = nn.LayerNorm(hid_dim) #second green box
        self.norm_ff  =  nn.LayerNorm(hid_dim) #third green box
        self.norm_maskedatt =  nn.LayerNorm(hid_dim) #first green box

        self.multi_masked  =  MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        # self.multi_cross   =  MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        self.ff      =  PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout =  nn.Dropout(dropout)
        
    def forward(self, trg, trg_mask):
        #trg: [b, tl, h]
        #enc_src: [b, sl, h]
        #trg_mask: [b, 1, tl, tl]
        #src_mask: [b, 1, 1, sl]
        
        #first block
        _trg, _ = self.multi_masked(trg, trg, trg, trg_mask) #q, k, v
        _trg    = self.dropout(_trg)
        _trg    = trg + _trg
        trg     = self.norm_maskedatt(_trg)
        
        # #second block
        # _trg, attention = self.multi_cross(trg, enc_src, enc_src, src_mask) #q, k, v
        # _trg    = self.dropout(_trg)
        # _trg    = trg + _trg
        # trg     = self.norm_att(_trg)
        
        #third block
        _trg    = self.ff(trg)
        _trg    = self.dropout(_trg)
        _trg    = trg + _trg
        trg     = self.norm_ff(_trg)
        
        return trg


class BeamSearchNode(object):
    def __init__(self,  previousNode, wordId, logProb, length):
        # self.h        = hiddenstate  #define the hidden state
        self.prevNode = previousNode  #where does it come from
        self.wordid   = wordId  #the numericalized integer of the word
        self.logp     = logProb  #the log probability
        self.len      = length  #the current length; first word starts at 1

    def eval(self, alpha=0.7):
        # the score will be simply the log probability penaltized by the length 
        # we add some small number to avoid division error
        # read https://arxiv.org/abs/1808.10006 to understand how alpha is selected
        return self.logp / float(self.len + 1e-6) ** (alpha)
    
    #this is the function for comparing between two beamsearchnodes, whether which one is better
    #it is called when you called "put"
    def __lt__(self, other):
        return self.len < other.len

    def __gt__(self, other):
        return self.len > other.len


class Decoder(nn.Module):
    
    def __init__(self, output_dim, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, device,trg_pad_idx, max_length = 130):
        super().__init__()
        self.trg_pad_idx = trg_pad_idx
        self.pos_emb = nn.Embedding(max_length, hid_dim)
        self.trg_emb = nn.Embedding(output_dim, hid_dim)
        self.scale   = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.dropout = nn.Dropout(dropout)
        self.layers  = nn.ModuleList(
                     [
                        DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                        for _ in range(n_layers)
                     ]
                     )
        self.fc      = nn.Linear(hid_dim, output_dim)
        self.device  = device
    
    def make_trg_mask(self, trg):
        
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_mask: [b, 1, 1, l]
        
        trg_len = trg_mask.shape[-1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        trg_mask = trg_mask & trg_sub_mask
        #trg_mask: [b, 1, l, l]
        
        return trg_mask

    def forward(self, trg):
        #trg = [b, tl]
        #enc_src = hidden states from encoder = [b, sl, h]
        #trg_mask = [b, 1, tl, tl]
        #src_mask = [b, 1, 1, sl]

        # create trg mask inside
        trg_mask = self.make_trg_mask(trg)

        b = trg.shape[0]
        l = trg.shape[1]
        
        #pos
        pos = torch.arange(0, l).unsqueeze(0).repeat(b, 1).to(self.device)
        #pos: [b, l]
        
        pos_emb = self.pos_emb(pos) #[b, l, h]
        trg_emb = self.trg_emb(trg) #[b, l, h]
        
        x = pos_emb + trg_emb * self.scale #[b, l, h]
        x = self.dropout(x)
        
        for layer in self.layers:
            trg = layer(x, trg_mask)

        # print(trg.shape)
        
        output = self.fc(trg)
        
        return output

    def decode(self, trg, method='beam-search'):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #src len = [batch size]

        # encoder_outputs, hidden = self.encoder(src, src_len) 
        #encoder_outputs = [src len, batch size, hid dim * 2]  (*2 because of bidirectional)(every hidden states)
        #hidden = [batch size, hid dim]  #final hidden state
       
        # hidden = hidden.unsqueeze(0)
        #hidden = [1, batch size, hid dim]
        
        if method == 'beam-search':
            return self.beam_decode(trg)
        else:
            return self.greedy_decode(trg)


    def greedy_decode(self, trg):

        prediction= self.forward(trg)
        prediction = prediction.squeeze(0)
        prediction = prediction # not include first one? 
        prediction = prediction.argmax(1) 

        return prediction


    def beam_decode(self,target_tensor):
        # remove ---
        #src_tensor      = [src len, batch size]---
        #target_tensor   = [trg len, batch size]
        #decoder_hiddens = [1, batch size, hid dim]---
        #encoder_outputs = [src len, batch size, hid dim * 2]---
        
        target_tensor = target_tensor.permute(1, 0)
        #target_tensor = [batch size, trg len]
        
        #how many parallel searches
        beam_width = 3
        
        #how many sentence do you want to generate
        topk = 1  
        
        #final generated sentence
        decoded_batch = []
                
        #Another difference is that beam_search_decoding has 
        #to be done sentence by sentence, thus the batch size is indexed and reduced to only 1.  
        #To keep the dimension same, we unsqueeze 1 dimension for the batch size.
        for idx in range(target_tensor.size(0)):  # batch_size
            
            #decoder_hiddens = [1, batch size, dec hid dim]
            # decoder_hidden = decoder_hiddens[:, idx, :]
            #decoder_hidden = [1, dec hid dim]

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([SOS_IDX]).to(device)

            # Number of sentence to generate
            endnodes = []  #hold the nodes of EOS, so we can backtrack
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode( None, decoder_input, 0, 1)
            nodes = PriorityQueue()  #this is a min-heap

            # start the queue
            nodes.put((-node.eval(), node))  #we need to put - because PriorityQueue is a min-heap
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                # score is log p divides by the length scaled by some constants
                score, n = nodes.get()
          
                # wordid is simply the numercalized integer of the word
                decoder_input  = n.wordid
                # decoder_hidden = n.h

                if n.wordid.item() == EOS_IDX and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                # decoder_input = SOS_IDX
                # decoder_hidden = [1, hid dim]


                prediction = self.forward(decoder_input.reshape(1,-1))
                #prediction     = [1, output dim]  #1 because the batch size is 1
                prediction = prediction.squeeze(0)

                #so basically prediction is probabilities across all possible vocab
                #we gonna retrieve k top probabilities (which is defined by beam_width) and their indexes
                #recall that beam_width defines how many parallel searches we want
                log_prob, indexes = torch.topk(prediction, beam_width)

                # log_prob      = (1, beam width)
                # indexes       = (1, beam width)
                
                nextnodes = []  #the next possible node you can move to

                # we only select beam_width amount of nextnodes
                for top in range(beam_width):
                    pred_t = indexes[0, top].reshape(-1)  #reshape because wordid is assume to be []; see when we define SOS
                    log_p = log_prob[0, top].item()
                                    
                    #decoder hidden, previous node, current node, prob, length
                    node = BeamSearchNode( n, pred_t, n.logp + log_p, n.len + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # Once everything is finished, choose nbest paths, back trace them
            
            ## in case it does not finish, we simply get couple of nodes with highest probability
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            #look from the end and go back....
            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace by looking at the previous nodes.....
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]  #reverse it....
                utterances.append(utterance) #append to the list of sentences....

            decoded_batch.append(utterances)

        return decoded_batch  #(batch size, length)
        