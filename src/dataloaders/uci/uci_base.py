"""UCI dataset"""
import torch.utils.data


class UCIData(torch.utils.data.Dataset):
    """UCI base class"""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path.expanduser()
