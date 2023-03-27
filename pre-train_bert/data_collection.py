import datasets
from tqdm.auto import tqdm


def collect_data():
    ds = datasets.load_dataset('oscar', 'unshuffled_deduplicated_la')
    text_data = []
    file_count = 0
    for sample in tqdm(ds['train']):
        sample = sample['text'].replace('\n', '')
        text_data.append(sample)
        if len(text_data) == 6000:
            with open(f'../data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1
        with open(f'../data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
    return None
