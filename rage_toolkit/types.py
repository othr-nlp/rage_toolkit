from abc import ABC, abstractmethod


class AugmentedGenerationSystem(ABC):
    @abstractmethod
    def generate_answer(self, query_string: str, document_texts: list[str]) -> tuple[str, str]:
        pass

    def delete():
        pass

class LlmSystem(ABC):
    @abstractmethod
    def run_inference(self, prompt : str) -> str:
        pass

class PromptGenerator(ABC):
    @abstractmethod
    def generate_prompt(self, query_string:str, document_texts: list[str]) -> str:
        pass
