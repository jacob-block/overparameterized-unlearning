from abc import ABC, abstractmethod

class DataGenerator(ABC):
    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def train_initial_model(self):
        pass

    @abstractmethod
    def train_gt_model(self):
        pass

    @abstractmethod
    def save_all(self, output_dir: str):
        pass
