import dynet
import numpy as np
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
        self.embeddings = pc.add_lookup_parameters((vocab_size, embed_size), dynet.UniformInitializer(0.1), name='embedding.weight')
        self.encoder = dynet.BiRNNBuilder(num_layers=1,
                                          input_dim=embed_size,
                                          hidden_dim=hidden_size,
                                          model=pc,
                                          rnn_builder_factory=dynet.SimpleRNNBuilder)
        self.decoder = dynet.SimpleRNNBuilder(layers=1,
                                              input_dim=embed_size,
                                              hidden_dim=hidden_size,
                                              model=pc)
        self.projection_layer_weights = pc.add_parameters((hidden_size, hidden_size * 2), dynet.UniformInitializer(0.1), name='projection.weight')
        self.projection_layer_bias = pc.add_parameters((hidden_size), dynet.UniformInitializer(0.1), name='projection.bias')
        self.output_layer_weights = pc.add_parameters((vocab_size, hidden_size), dynet.UniformInitializer(0.1), name='output.weight')
        self.output_layer_bias = pc.add_parameters((vocab_size), dynet.UniformInitializer(0.1), name='output.bias')

        for params in self.encoder.param_collection().parameters_list():
            params.set_value(np.random.uniform(-0.1, 0.1, params.shape()))
        for params in self.decoder.get_parameters()[0]:
            params.set_value(np.random.uniform(-0.1, 0.1, params.shape()))

    def calculate_projection(self,
                             target_encoding: Expression,
                             context: Expression) -> List[Expression]:
        W = dynet.parameter(self.projection_layer_weights)
        b = dynet.parameter(self.projection_layer_bias)
        concat = dynet.concatenate([target_encoding, dynet.transpose(context)])
        return dynet.affine_transform([b, W, concat])

    def calculate_output_scores(self,
                                hidden: Expression) -> List[Expression]:
        W = dynet.parameter(self.output_layer_weights)
        b = dynet.parameter(self.output_layer_bias)
        return dynet.transpose(dynet.affine_transform([b, W, hidden]))

    def calculate_loss(self,
                       target_tokens: List[int],
                       scores: List[Expression]) -> Expression:
        losses = []
        for score, token in zip(scores, target_tokens):
            log_probs = dynet.log_softmax(score)
            losses.append(-log_probs[token])
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

        source_encoding = dynet.concatenate_cols(source_encoding)
        target_encoding = dynet.concatenate_cols(target_encoding)

        affinities = dynet.transpose(target_encoding) * source_encoding
        context = affinities * dynet.transpose(source_encoding)

        projection = self.calculate_projection(target_encoding, context)
        scores = self.calculate_output_scores(projection)

        target_tokens = target[1:]
        scores = scores[:-1]

        return self.calculate_loss(target_tokens, scores)


def print_param_info(pc: ParameterCollection) -> None:
    for param in pc.lookup_parameters_list() + pc.parameters_list():
        norm = np.linalg.norm(param.as_array())
        grad_norm = np.linalg.norm(param.grad_as_array())
        print(f'{param.name()}\t{norm}\t{grad_norm}')


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
