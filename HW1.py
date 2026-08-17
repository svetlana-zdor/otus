from collections import defaultdict

class TestLLM:
    def __init__(self):
        # Словарь для хранения частот: { текущее_слово: { следующее_слово: количество } }
        self.ngram_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.train_data: list[str] = []
    
    def train(self, data: list[str]) -> None:
        """
        Метод обучения модели.
        Проходит по всем предложениям, разбивает их на слова 
        и записывает частоту следования одного слова за другим.
        """
        self.train_data = data
        
        for sentence in data:
            # Разбиваем предложение на слова по пробелам
            words: list[str] = sentence.split()
            
            # Проходим по всем словам в предложении, кроме последнего (так как за ним нет следующего)
            for i in range(len(words) - 1):
                current_word: str = words[i]
                next_word: str = words[i + 1]
                
                # Увеличиваем счетчик для пары (текущее_слово, следующее_слово)
                self.ngram_counts[current_word][next_word] += 1

    def predict_next_word(self, start_word: str) -> str:
        if start_word not in self.ngram_counts:
            return "Слово не найдено в обучающей выборке."
        
        next_words: defaultdict[str, int] = self.ngram_counts[start_word]
        
        if not next_words:
            return "Нет данных для предсказания."
        
        # Находим слово с максимальной частотой (значением в словаре)
        most_frequent: str = max(next_words, key=next_words.get)
        
        return most_frequent

# Исходные данные для обучения
data: list[str] = [
    "кот спит на диване",
    "кот ест рыбу",
    "кот играет с мячом",
    "кот спит на окне",
    "кот гуляет по улице",
    "кот ест молоко",
    "кот играет с мышкой",
    "кот спит в коробке",
    "кот смотрит в окно",
    "кот гуляет в парке"
]

# Инициализация и обучение
test_llm_model: TestLLM = TestLLM()
test_llm_model.train(data)

# Тестирование
test_word: str = "кот"
prediction_word: str = test_llm_model.predict_next_word(test_word)

print(f"Входное слово: '{test_word}'")
print(f"Предсказанное следующее слово: '{prediction_word}'")