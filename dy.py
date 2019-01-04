import dynet
from dynet import Expression, ParameterCollection, RNNState
from tqdm import tqdm
from typing import List, Tuple

from util import load_data, load_vocab, get_vocab_size


class DyNetModel(object):
    def __init__(self,
                 pc: ParameterCollection,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int) -> None:
        self.embeddings = pc.add_lookup_parameters((vocab_size, embed_size))
        self.encoder = dynet.BiRNNBuilder(num_layers=1,
                                          input_dim=embed_size,
                                          hidden_dim=hidden_size,
                                          model=pc,
                                          rnn_builder_factory=dynet.SimpleRNNBuilder)
        self.decoder = dynet.SimpleRNNBuilder(layers=1,
                                              input_dim=embed_size,
                                              hidden_dim=hidden_size,
                                              model=pc)
        self.attention_W_weights = pc.add_parameters((hidden_size, hidden_size * 2))
        self.attention_W_bias = pc.add_parameters((hidden_size))
        self.attention_v_weights = pc.add_parameters((hidden_size))
        self.projection_layer_weights = pc.add_parameters((hidden_size, hidden_size * 2))
        self.projection_layer_bias = pc.add_parameters((hidden_size))
        self.output_layer_weights = pc.add_parameters((vocab_size, hidden_size))
        self.output_layer_bias = pc.add_parameters((vocab_size))

    def calculate_attention(self,
                            source_encoding: List[Expression],
                            target_encoding: List[Expression]) -> List[Expression]:
        W = dynet.parameter(self.attention_W_weights)
        b = dynet.parameter(self.attention_W_bias)
        v = dynet.parameter(self.attention_v_weights)
        affinities_matrix = []
        for target_state in target_encoding:
            affinities = []
            for source_state in source_encoding:
                concat = dynet.concatenate([source_state, target_state])
                affinity = dynet.transpose(v) * dynet.tanh(dynet.affine_transform([b, W, concat]))
                affinities.append(affinity)
            affinities_matrix.append(dynet.concatenate(affinities))
        return affinities_matrix

    def calculate_context(self,
                          source_encoding: List[Expression],
                          affinity_matrix: List[Expression]) -> List[Expression]:
        contexts = []
        for affinities in affinity_matrix:
            attention = dynet.softmax(affinities)
            context = dynet.esum([dynet.cmult(state, prob) for state, prob in zip(source_encoding, attention)])
            contexts.append(context)
        return contexts

    def calculate_projection(self,
                             target_encoding: List[Expression],
                             contexts: List[Expression]) -> List[Expression]:
        W = dynet.parameter(self.projection_layer_weights)
        b = dynet.parameter(self.projection_layer_bias)
        projections = []
        for encoding, context in zip(target_encoding, contexts):
            projection = dynet.affine_transform([b, W, dynet.concatenate([encoding, context])])
            projections.append(projection)
        return projections

    def calculate_output_scores(self,
                                hidden: List[Expression]) -> List[Expression]:
        W = dynet.parameter(self.output_layer_weights)
        b = dynet.parameter(self.output_layer_bias)
        outputs = []
        for h in hidden:
            output = dynet.affine_transform([b, W, h])
            outputs.append(output)
        return outputs

    def calculate_loss(self,
                       target_tokens: List[int],
                       scores: List[Expression]) -> Expression:
        losses = []
        for score, token in zip(scores, target_tokens):
            probs = dynet.softmax(score)
            losses.append(-dynet.log(probs[token]))
        return dynet.esum(losses)

    def __call__(self,
                 source: List[int],
                 target: List[int]) -> Expression:
        source_embedding = [self.embeddings[x] for x in source]
        source_encoding = self.encoder.add_inputs(source_embedding)
        hidden = dynet.concatenate([source_encoding[-1][0].s()[0], source_encoding[-1][1].s()[0]])

        target_embedding = [self.embeddings[x] for x in target]
        decoder_state = self.decoder.initial_state().set_s((hidden,))
        target_encoding = decoder_state.add_inputs(target_embedding)

        source_encoding = [dynet.concatenate([state[0].output(), state[1].output()])
                           for state in source_encoding]
        target_encoding = [state.output() for state in target_encoding]

        affinities = self.calculate_attention(source_encoding, target_encoding)
        context = self.calculate_context(source_encoding, affinities)

        projection = self.calculate_projection(target_encoding, context)
        scores = self.calculate_output_scores(projection)

        target_tokens = target[1:]
        scores = scores[:-1]

        return self.calculate_loss(target_tokens, scores)


def main():
    train = load_data('data/train.jsonl')
    token2id = load_vocab('data/vocab.json')
    vocab_size = get_vocab_size(token2id)
    embed_size = 20
    hidden_size = 40
    num_epochs = 15

    pc = dynet.ParameterCollection()
    model = DyNetModel(pc, vocab_size, embed_size, hidden_size)
    optimizer = dynet.SimpleSGDTrainer(pc, 0.1)

    for _ in range(num_epochs):
        total_loss = 0
        with tqdm(train) as pbar:
            for i, instance in enumerate(pbar):
                source = instance['source']
                target = instance['target']
                loss = model(source, target)

                total_loss += loss.value()
                average_loss = total_loss / (i + 1)
                loss_str = f'{average_loss:.4f}'
                pbar.set_description(loss_str)

                loss.backward()
                optimizer.update()
                dynet.renew_cg()


if __name__ == '__main__':
    main()
