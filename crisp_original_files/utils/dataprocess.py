import json
import os
from abc import ABC, abstractmethod


class DataProcess(ABC):
    """
    抽象基类，用于定义数据处理的基础接口。
    """
    def __init__(self, dataset_name, split='dev', base_path="./data"):
        """
        初始化数据处理器。
        :param dataset_name: 数据集名称
        :param split: 数据集的分割 (e.g., 'train', 'test', 'validation')
        :param base_path: 数据集存储的基础路径
        """
        self.dataset_name = dataset_name
        self.split = split
        self.base_path = base_path
        self.file_path = os.path.join(base_path, dataset_name, f"{split}.json")
        self.data = None
        self.read_data()

    def read_data(self):
        """
        读取 JSON 文件中的数据。
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.file_path}")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

    @abstractmethod
    def extract_content(self):
        """
        抽象方法，提取需要的内容，子类需实现。
        """
        pass

    @abstractmethod
    def extract_prediction(self, model_response):
        """
        抽象方法，从模型的回答中提取预测答案。
        """
        pass




class DataProcessorFactory:
    """
    工厂类，用于根据数据集名称动态实例化数据处理类。
    """
    @staticmethod
    def create_processor(dataset_name, split, base_path="./data"):
        """
        根据数据集名称返回对应的处理器实例。
        :param dataset_name: 数据集名称
        :param split: 数据集分割 (e.g., 'train', 'test', 'validation')
        :param base_path: 数据集存储路径
        :return: 对应的 DataProcess 子类实例
        """
        if dataset_name == "example_dataset":
            return JSONDataProcess(dataset_name, split, keys=("question", "label", "reason"), base_path=base_path)
        elif dataset_name == "qa_dataset":
            return QADataProcess(dataset_name, split, keys=("question", "label", "reason"), base_path=base_path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")