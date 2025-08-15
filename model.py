import torch
import torch.nn as nn


class CnnModel(nn.Module):
    def __init__(self, in_channels):
        super(CnnModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
        )
        self.cnn_output = None

    def train_model(self, input_data, train_info, epochs, lr: float = 0.004):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = self.cnn(input_data)
            total_loss = self.compute_loss(embeddings, train_info)

            with torch.no_grad():
                print(total_loss.item())

            total_loss.backward()
            optimizer.step()

    def predict(self, samples: torch.Tensor):
        self.eval()
        with torch.no_grad():
            embeddings = self.cnn(samples)
            return torch.norm(embeddings[0] - embeddings[1], p=2).item()


    @staticmethod
    def compute_loss(embeddings, samples_info):
        samples_info = torch.from_numpy(samples_info)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # shape (B, B)

        labels = samples_info[:, 0].unsqueeze(1) == samples_info[:, 0].unsqueeze(0)
        active_mask = (samples_info[:, 1] == 1).unsqueeze(1) & (samples_info[:, 1] == 1).unsqueeze(0)
        positive_mask = labels & active_mask
        negative_mask = ~labels & active_mask

        eye = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        positive_mask = positive_mask & ~eye
        negative_mask = negative_mask & ~eye

        pos_loss = dist_matrix[positive_mask].pow(2)
        neg_loss = torch.relu(1.0 - dist_matrix[negative_mask]).pow(2)

        total_loss = (pos_loss.sum() + neg_loss.sum()) / (len(pos_loss) + len(neg_loss) + 1e-8)
        return total_loss

    # @staticmethod
    # def generate_pairs(samples, samples_info):
    #     pairs = []
    #     labels = []
    #     for i, seq1 in enumerate(samples):
    #         for j, seq2 in enumerate(samples):
    #             if i != j:
    #                 pairs.append((seq1, seq2))
    #                 if (samples_info[j, 0] == samples_info[i, 0]
    #                         and samples_info[j, 1] == 1
    #                         and samples_info[i, 1] == 1):
    #                     y = 1
    #                 else:
    #                     y = 0
    #                 labels.append(torch.tensor(y, dtype=torch.float32, device=samples.device))
    #     return pairs, labels

    @staticmethod
    def contrastive_loss(seq1, seq2, y):
        distance = (seq1 - seq2).pow(2).sum().sqrt()
        return y*distance + (1 - y)*torch.relu(1 - distance)

