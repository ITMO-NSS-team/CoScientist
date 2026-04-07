from abc import ABC, abstractmethod


class ArticleIdentityResolver(ABC):
    
    @abstractmethod
    def compute_id(self, raw_data) -> str:
        pass
        
    @abstractmethod
    def is_duplicate(self, article_id) -> bool:
        pass
    