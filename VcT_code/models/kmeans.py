import torch

def pairwise_distance(X, Y):
    # X shape: (batch_size, num_points, channels)
    # Y shape: (batch_size, num_clusters, channels)
    A = X.unsqueeze(2)  # Shape: (batch_size, num_points, 1, channels)
    B = Y.unsqueeze(1)  # Shape: (batch_size, 1, num_clusters, channels)
    dis = (A - B) ** 2.0
    return dis.sum(dim=-1)  # Shape: (batch_size, num_points, num_clusters)

def kmeans(X, num_clusters, max_iter=100):
    batch_size, num_points, channels = X.shape
    initial_state = X[:, torch.randperm(num_points)[:num_clusters], :]  # Shape: (batch_size, num_clusters, channels)

    for _ in range(max_iter):
        dis = pairwise_distance(X, initial_state)  # Shape: (batch_size, num_points, num_clusters)
        choice_cluster = torch.argmin(dis, dim=2)  # Shape: (batch_size, num_points)

        # Compute new cluster centers
        new_state = torch.zeros((batch_size, num_clusters, channels), device=X.device)
        for b in range(batch_size):
            for c in range(num_clusters):
                mask = choice_cluster[b] == c
                if mask.any():
                    new_state[b, c] = X[b, mask].mean(dim=0)

        if torch.allclose(new_state, initial_state, atol=1e-4):
            break

        initial_state = new_state

    return choice_cluster, initial_state  # choice_cluster: (batch_size, num_points), initial_state: (batch_size, num_clusters, channels)
