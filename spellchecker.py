import re
import collections
from itertools import product
import functools
import os


# будем ли мы брать только самый вероятный вариант или перебирать все возможные
# Если False, то парсер быстрее, но менее аккуратный
SHORT_CIRCUIT = False
# это просто гласные и весь наш алфавит
vowels = set('аеёиоуыэюя')
alphabet = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

# два вида сравнения, которые понадобятся в ходе отбора кандидатов
def cmp2(a, b):
    return (a > b) - (a < b)


def cmp(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0

def words(text):
    """ находим в тексте все слова """
    return re.findall('[а-я]+', text.lower())

def train(text, model=None):
    """ сгенерировать или обновить частотную модель (словарь вида word:frequency) """
    model = collections.defaultdict(lambda: 0) if model is None else model
    for word in words(text):
        model[word] += 1
    return model

def train_from_files(file_list, model=None):
    """ читаем все файлы в корпусе по порядку и обрабатываем слова в них"""
    for f in file_list:
        try:
            model = train(open(f).read(), model)
        except:
            continue
    return model

# А тут всякие вспомогательные функции

def numberofdupes(string, idx):
    """ ищем, дублируется ли символ """
    # "abccdefgh", 2  returns 1
    initial_idx = idx
    last = string[idx]
    while idx+1 < len(string) and string[idx+1] == last:
        idx += 1
    return idx-initial_idx

def hamming_distance(word1, word2):
    """ считаем расстояние Хэмминга между двумя кандидатами """
    if word1 == word2:
        return 0
    dist = sum(map(str.__ne__, word1[:len(word2)], word2[:len(word1)]))
    dist = dist+abs(len(word2)-len(word1)) # max([word2, word1]) if not dist else
    return dist

def frequency(word, word_model):
    """ извлекаем частотность слова в модели """
    return word_model.get(word, 0)

# тут всякие функции, которые анализируют вероятность появления в тексте того или иного кандидата

def variants(word):
    """ тут мы находим все строки, которые находятся на расстоянии 1 правки от интересующего нас слова """
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts    = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def double_variants(word):
    """ а тут на расстоянии двух правок """
    return set(s for w in variants(word) for s in variants(w))

def reductions(word):
    """ а тут какие слова могут получиться, если из нашего слова удалим символы-дубликаты """
    word = list(word)
    # ['д','а', 'а', 'а'] --> ['д', ['а', 'аа', 'ааа']]
    for idx, l in enumerate(word):
        n = numberofdupes(word, idx)
        if n:
            flat_dupes = [l*(r+1) for r in range(n+1)][:3] # только 3 повтора подряд учитываются
            for _ in range(n):
                word.pop(idx+1)
            word[idx] = flat_dupes

    # ['д',['а','аа','ааа']] --> 'да','даа','дааа'
    for p in product(*word):
        yield ''.join(p)

def vowelswaps(word):
    """ а тут все кандидаты, которые могут получиться при замене гласных """
    word = list(word)
    # ['д','а'] --> ['д', ['а', 'е', 'и', 'у', 'ы', ...]]
    for idx, l in enumerate(word):
        if type(l) == list:
            pass                       
        elif l in vowels:
            word[idx] = list(vowels)    
    i = 0
    for p in product(*word):
        if i == 100:
            break
        yield ''.join(p)
        i += 1

def both(word):
    """ а тут совмещаем удаление символа с заменой согласного """
    for reduction in reductions(word):
        for variant in vowelswaps(reduction):
            yield variant

# выбираем кандидата

def suggestions(word, real_words, short_circuit=True):
    """ выбираем возможные исправления, полученные при помощи различных операций """
    word = word.lower()
    if short_circuit:  # перебирать все варианты или только самый вероятный
        return ({word}                      & real_words or   #  caps
                set(reductions(word))       & real_words or   #  repeats
                set(vowelswaps(word))       & real_words or   #  vowels
                set(variants(word))         & real_words or   #  other
                set(both(word))             & real_words or   #  both
                set(double_variants(word))  & real_words or   #  other
                {word})
    else:
        return ({word}                      & real_words or
                (set(reductions(word))  | set(vowelswaps(word)) | set(variants(word)) | set(both(word)) | set(double_variants(word))) & real_words or
                {word})

def best(inputted_word, suggestions, word_model=None):
    """ в созданном листе исправлений выбираем лучшее, основываясь либо на расстоянии Хэмминга,
    либо на частотности слов в текстовой модели """

    suggestions = list(suggestions)

    def comparehamm(one, two):
        score1 = hamming_distance(inputted_word, one)
        score2 = hamming_distance(inputted_word, two)
        return cmp2(score1, score2)  # lower is better

    def comparefreq(one, two):
        score1 = frequency(one, word_model)
        score2 = frequency(two, word_model)
        return cmp2(score2, score1)  # higher is better

    freq_sorted = sorted(suggestions, key=functools.cmp_to_key(comparefreq))[:10]     # take the top 10
    hamming_sorted = sorted(suggestions, key=functools.cmp_to_key(comparehamm))[:10]  # take the top 10
    return freq_sorted[0]

def process_text(f_in, f_out):
    """ Ура! Обрабатываем текст!!! """
    n = open(f_out, 'w')
    with open(f_in) as f:
        for line in f:
            new_line = []
            for word in line.strip().split():
                results = suggestions(word, real_words, SHORT_CIRCUIT)
                new_line.append(best(word, results, word_model))
            print(' '.join(new_line))
            n.write(' '.join(new_line) + '\n')
    n.close()

if __name__ == '__main__':
    # инициализируем частотную модель с помощью простого списка слов
    print('Initializing model...')
    word_model = train(open('1grams-3.txt').read())
    real_words = set(word_model)

    # теперь дообучаем модель с новыми текстами, из которых она возьмет частотности слов
    print('Training with natural texts...')
    files_dir = ''
    texts = []
    for root, dirs, files in os.walk('.'):
        texts += [os.path.join(root, f) for f in files]
    
    word_model = train_from_files(texts, word_model)

    process_text('test_dialogue.txt', 'corrected_dialogue.txt')
