"""Training-free answer-frequency baseline used in the validation comparison."""
from collections import Counter
from common import parse_answer


def predict(bank, arrays, checkpoint):
    choices = []
    for row in bank:
        answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
        counts = Counter(answer for answer in answers if answer is not None)
        choices.append(max(range(4), key=lambda i: counts.get(answers[i], 0)) if counts else 0)
    return choices
