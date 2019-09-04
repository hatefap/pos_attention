from util import Bijenkhan

bijenkhan = Bijenkhan('/home/hatef/courses/Term-2/NLP/HWs/CA4/train_simple.txt')
gen = bijenkhan.sent_tag_gen(100)
s, t = next(gen)
print(bijenkhan.get_vocab())
print(bijenkhan.get_tags())
