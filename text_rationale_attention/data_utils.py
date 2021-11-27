
import random, pickle

import numpy as np

from latent_rationale.beer.util import beer_reader

def split_and_save(fn, pos_scores, neg_scores, num_train, num_val, num_test, out_fn):
    examples = beer_reader(fn, pos_scores, neg_scores, aspect=0, max_len=256)
    pos_examples = [(e.tokens, e.label) for e in examples if e.label == 1]
    neg_examples = [(e.tokens, e.label) for e in examples if e.label == 0]
    assert len(pos_examples) >= (num_train + num_val + num_test) / 2
    assert len(neg_examples) >= (num_train + num_val + num_test) / 2
    random.shuffle(pos_examples)
    random.shuffle(neg_examples)
    train = pos_examples[0 : num_train // 2] + neg_examples[0 : num_train // 2]
    val = pos_examples[num_train // 2 : num_train // 2 + num_val // 2] + neg_examples[num_train // 2 : num_train // 2 + num_val // 2]
    test = pos_examples[num_train // 2 + num_val // 2 : num_train // 2 + num_val // 2 + num_test // 2] + \
           neg_examples[num_train // 2 + num_val // 2 : num_train // 2 + num_val // 2 + num_test // 2]
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    data = {'train': train, 'val': val, 'test': test}
    pickle.dump(data, open(out_fn, 'wb'))

def visualize_data(fn):
    data = pickle.load(open(fn, 'rb'))
    train_x, train_y = zip(*data['train'])
    val_x, val_y = zip(*data['val'])
    test_x, test_y = zip(*data['test'])
    print(np.array(train_y).mean(), np.array(val_y).mean(), np.array(test_y).mean())
    for x, y in zip(train_x, train_y):
        print(['NEGATIVE', 'POSITIVE'][y], ' '.join(x))
        input()

def load_data(fn):
    data = pickle.load(open(fn, 'rb'))
    return data

if __name__ == '__main__':
    # split_and_save('data/beer/reviews.aspect0.train.txt.gz', [0.9], [0.1, 0.2, 0.3], 10000, 1000, 1000, 'binary_distinct_123_9.pkl')
    # split_and_save('data/beer/reviews.aspect0.train.txt.gz', [0.6], [0.5], 10000, 1000, 1000, 'binary_confusing_5_6.pkl')
    visualize_data('binary_distinct_123_9_article_m0.5.pkl')
    # visualize_data('binary_confusing_5_6.pkl')
    pass
