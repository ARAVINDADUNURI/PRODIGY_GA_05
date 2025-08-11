import random
from collections import defaultdict

class MarkovChainTextGenerator:
    def __init__(self):
        self.model = defaultdict(list)

    def train(self, text):
        """Builds the Markov model from the input text."""
        words = text.split()
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            self.model[current_word].append(next_word)

    def generate_text(self, start_word=None, length=50):
        """Generates text using the trained Markov model."""
        if not self.model:
            raise Exception("The model hasn't been trained yet.")

        if start_word is None or start_word not in self.model:
            start_word = random.choice(list(self.model.keys()))

        result = [start_word]
        current_word = start_word

        for _ in range(length - 1):
            next_words = self.model.get(current_word)
            if not next_words:
                break  
            next_word = random.choice(next_words)
            result.append(next_word)
            current_word = next_word

        return ' '.join(result)


# Example usage:
if __name__ == "__main__":
    input_text = """
    Artificial intelligence is the simulation of human intelligence processes by machines,
    especially computer systems. These processes include learning, reasoning, and self-correction.
    Applications of AI include expert systems, natural language processing, and machine vision.
    """

    generator = MarkovChainTextGenerator()
    generator.train(input_text)

    print("\n--- Generated Text ---\n")
    print(generator.generate_text(length=30))
