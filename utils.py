import torch

def decode_sequence(tokenizer, sequences):
    return tokenizer.decode(sequences[0])

def get_hidden_embeddings(hidden_states):
    _start = torch.squeeze(torch.stack(hidden_states[0]).transpose(0,2), dim=1)
    _hs = torch.stack([torch.reshape(torch.stack(x), [13, 768]) for x in hidden_states[1:]])
    
    return torch.concat([_start, _hs])

def get_n_layer_hidden_embeddings(hidden_states, layer_n=12):
    # Default to getting the last layer (12)
    
    # The start sequence embeddings are in the first tuple element
    _start = torch.flatten(hidden_states[0][layer_n], start_dim=-1)
    
    # The rest of the sequence embeddings are obtained
    _hs = [x[layer_n] for x in hidden_states[1:]]
    _hs = torch.concat(_hs, dim=1)
    
    return torch.concat([_start, _hs], dim=1).squeeze(dim=0)


def filter_token_length(sequences, tokenizer, tok_len=19):
    filtered_sequences = []
    for seq in sequences:
        seq_ids = tokenizer.encode(seq, return_tensors='pt')
        if len(seq_ids[0]) < 20:
            filtered_sequences.append(seq)
    return filtered_sequences