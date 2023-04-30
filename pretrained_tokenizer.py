from transformers import BertTokenizer

text = 'happy, pronounce, sadly, disadvantage, incorrect, precook, cook, prehistorical, underage, Superclass'
prefix_stem = 'undesirable, desirable, unhappy, happy, correct, incorrect, ability, disability, pulling, dishwasher'
ing_words = 'being, pulling, creating, undertaking, streaming discerning'
ed_words = 'handed, looked discerned'
anti_words = 'antisocial social antigravity gravity antifreeze'
filtered_anti_words = 'anti social social anti gravity gravity anti freeze'
dis_words = 'dislike like dishonest honest dishwasher disagree agree'
filtered_dis_words = 'dis like like dis honest honest dishwasher dis agree agree'
uni_words = 'unicycle universal unilateral unanimous uni'
filtered_uni_words = 'uni cycle uni versal uni lateral unanimous discern distinguish discernment undesirable unrelated cute '
num = '1, 10, 100, 102, 1000, 10000, 1002, 10000000, 200, 201, 333, 3333'
test_2 = 'a complaint about your disruptive behavior here: https : / / en . wikipedia . org / wiki / wikipedia : ' \
         'administrators % 27 noticeboard / incidents # disruptive users vandalizing article about spiro koleka'
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, padding="max_length",
                                     return_attention_mask=True, return_tensors="pt")
    tokens = tokenizer.tokenize('rumination, unilateral, undesirable, antisocial')
    print(tokens)
