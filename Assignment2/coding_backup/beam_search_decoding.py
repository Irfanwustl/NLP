import math
def beam_search_decoding(encoded, k):
  '''
  args:
  encoded: a sequence of probabilities of a vocabulary with size of [[prob]*length of dictionary]*length of sequence

  return:
  a sequence of sequence of the index of the the decoded word in dictionary. size= [[index]*length of sentence]*k
  '''
  #your answer here
  output_sequences = []
  for i,next_word_probs in enumerate(encoded):
        if i==0:
            queue = [(0,[])]
        children = []
        for current_state in queue:
            for word_index,word_prob in enumerate(next_word_probs):
                next_state_logprob = current_state[0]+math.log(word_prob)
                next_state_sentence = current_state[1].copy()
                next_state_sentence.append(word_index)
                children.append((next_state_logprob,next_state_sentence))
                
        queue = sorted(children,reverse=True)[:k]
       
                
                
  for sentence in queue:
    output_sequences.append(sentence[1])
        

  return np.array(output_sequences)
