import numpy as np

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        """Store data in a flat contiguous buffer simulating shared memory."""
        self.n_samples, self.n_features = data.shape
        self.batch_size = batch_size
        self.buffer = np.ascontiguousarray(data).ravel()

    def num_batches(self) -> int:
        """Return total number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def get_batch(self, batch_idx: int) -> np.ndarray:
        """Return batch as a zero-copy view into the buffer."""
        if batch_idx < 0 or batch_idx >= self.num_batches():
            raise IndexError("Invalid batch index")

        start = batch_idx * self.batch_size * self.n_features
        end = min((batch_idx + 1) * self.batch_size,
                  self.n_samples) * self.n_features

        return self.buffer[start:end].reshape(-1, self.n_features)

    def is_zero_copy(self, batch_idx: int) -> bool:
        """Check whether the batch shares memory with the buffer."""
        batch = self.get_batch(batch_idx)
        return np.shares_memory(batch, self.buffer)

    def get_batch_means(self) -> list:
        """Return list of per-batch mean values, each rounded to 4 decimals."""
        return [
            round(float(self.get_batch(i).mean()), 4)
            for i in range(self.num_batches())
        ]

    def write_to_buffer(self, row: int, col: int, value: float) -> None:
        """Write a value directly into the flat buffer at (row, col)."""
        if not (0 <= row < self.n_samples and
                0 <= col < self.n_features):
            raise IndexError("Invalid row or column")

        self.buffer[row * self.n_features + col] = value