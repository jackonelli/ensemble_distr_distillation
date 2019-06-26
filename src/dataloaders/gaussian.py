from pathlib import Path
import csv
import numpy as np
import torch.utils.data


class SyntheticGaussianData(torch.utils.data.Dataset):
    def __init__(self,
                 mean_0,
                 cov_0,
                 mean_1,
                 cov_1,
                 store_file,
                 reuse_data=False,
                 n_samples=1000,
                 ratio_0_to_1=0.5):
        super(SyntheticGaussianData).__init__()
        self.mean_0 = np.array(mean_0)
        self.mean_1 = np.array(mean_1)
        self.cov_0 = np.array(cov_0)
        self.cov_1 = np.array(cov_1)
        self.n_samples = n_samples
        self.ratio_0_to_1 = ratio_0_to_1
        self.file = Path(store_file)
        if self.file.exists() and reuse_data:
            self.validate_dataset()
        else:
            print("Sampling new data")
            self.sample_new_data()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = None
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            for count, row in enumerate(csv_reader):
                if count == index:
                    sample = row
                    break

        return np.array(sample, dtype=float)

    def sample_new_data(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        size_0 = int(np.floor(self.n_samples * self.ratio_0_to_1))
        size_1 = self.n_samples - size_0
        sampled_x_0 = np.random.multivariate_normal(mean=self.mean_0,
                                                    cov=self.cov_0,
                                                    size=size_0)
        y_0 = np.zeros((size_0, 1))
        sampled_x_1 = np.random.multivariate_normal(mean=self.mean_1,
                                                    cov=self.cov_1,
                                                    size=size_1)
        y_1 = np.ones((size_1, 1))

        all_x = np.row_stack((sampled_x_0, sampled_x_1))
        all_y = np.row_stack((y_0, y_1))
        combined_data = np.column_stack((all_x, all_y))
        np.savetxt(self.file, combined_data, delimiter=",")

    def validate_dataset(self):
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            assert len(self.mean_0) == len(next(csv_reader)) - 1
            assert self.n_samples - 1 == sum(1 for row in csv_reader)

    def get_full_data(self):
        tmp_raw_data = list()
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            tmp_raw_data = [data for data in csv_reader]
        return np.array(tmp_raw_data, dtype=float)
