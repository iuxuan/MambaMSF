from dataclasses import dataclass
from typing import List

@dataclass
class TrainingConfig:
    # Dataset
    dataset_index: int = 2
    data_set_path: str = './data'
    data_set_name_list: List[str] = ('UP', 'HanChuan', 'HongHu')
    #                                 0      1           2       
    # Parameters
    work_dir: str = './'
    exp_name: str = 'RUNS'
    net_name: str = 'MambaMSF'
    lr: float = 0.0001
    max_epoch: int = 200
    train_samples: int = 30
    val_samples: int = 10
    hidden_dim: int = 128
    record_computecost: bool = True
    flag_list: List[int] = (1, 0) 
    ratio_list: List[float] = (0.00, 0.00) 
    device: str = "cuda:0"
    seed_list: List[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    def __post_init__(self):
        self._split_image = None

    @property
    def data_set_name(self) -> str:
        return self.data_set_name_list[self.dataset_index]

    def get_save_folder(self) -> str:
        return f"{self.work_dir}/{self.exp_name}/{self.net_name}/{self.data_set_name}"

    @property
    def split_image(self) -> bool:
        if self._split_image is not None:
            return self._split_image
        return self.data_set_name in ['HanChuan', 'Houston']
    
    @split_image.setter
    def split_image(self, value: bool):
        self._split_image = value