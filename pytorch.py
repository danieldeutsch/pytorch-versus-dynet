import torch
from tqdm import tqdm

from util import load_data, load_vocab, get_vocab_size


class PyTorchModel(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = torch.nn.Embedding(vocab_size, embed_size)
        self.encoder = torch.nn.RNN(input_size=embed_size,
                                    hidden_size=hidden_size // 2,
                                    batch_first=True,
                                    bidirectional=True)
        self.decoder = torch.nn.RNN(input_size=embed_size,
                                    hidden_size=hidden_size,
                                    batch_first=True)
        self.attention_W = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.attention_v = torch.nn.Linear(hidden_size, 1)
        self.projection_layer = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, vocab_size)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def calculate_attention(self,
                            source_encoding: torch.Tensor,
                            target_encoding: torch.Tensor) -> torch.Tensor:
        num_source_tokens = source_encoding.size(1)
        num_target_tokens = target_encoding.size(1)

        source_encoding = source_encoding.unsqueeze(1)
        target_encoding = target_encoding.unsqueeze(2)

        source_encoding = source_encoding.expand(-1, num_target_tokens, -1, -1)
        target_encoding = target_encoding.expand(-1, -1, num_source_tokens, -1)

        concat = torch.cat([source_encoding, target_encoding], dim=-1)
        affinities = self.attention_v(torch.tanh(self.attention_W(concat))).squeeze(-1)
        return affinities

    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        source_embedding = self.embeddings(source)
        source_encoding, hidden = self.encoder(source_embedding)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1).unsqueeze(0)

        target_embedding = self.embeddings(target)
        target_encoding, _ = self.decoder(target_embedding, hidden)

        affinities = self.calculate_attention(source_encoding, target_encoding)
        attention = torch.nn.functional.softmax(affinities, dim=-1)
        context = torch.bmm(attention, source_encoding)

        projection = self.projection_layer(torch.cat([target_encoding, context], dim=-1))
        scores = self.output_layer(projection)

        target_tokens = target[:, 1:].view(-1)
        scores = scores[:, :-1].view(-1, self.vocab_size)

        return self.loss(scores, target_tokens)


def main():
    train = load_data('data/train.jsonl')
    token2id = load_vocab('data/vocab.json')
    vocab_size = get_vocab_size(token2id)
    embed_size = 20
    hidden_size = 40
    num_epochs = 15

    model = PyTorchModel(vocab_size, embed_size, hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(num_epochs):
        total_loss = 0
        with tqdm(train) as pbar:
            for i, instance in enumerate(pbar):
                source = torch.LongTensor(instance['source']).unsqueeze(0)
                target = torch.LongTensor(instance['target']).unsqueeze(0)
                loss = model(source, target)

                total_loss += loss.item()
                average_loss = total_loss / (i + 1)
                loss_str = f'{average_loss:.4f}'
                pbar.set_description(loss_str)

                tqdm.write(str(loss.item()))
                if torch.isnan(loss):
                    pbar.close()
                    exit()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    main()
