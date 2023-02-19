import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
print(sentence + " - ", similarity)

# What I find interesting about the similarities is that it is clear that not only is the characteristics of the object being considered but also the context of the object.
# For example both cat and monkey show higher similarities because they are both animals, and apple and banana show higher similarities because they are both fruits.
# The thing that becomes interesting is the effect of context on similarity. For example, monkey and banana scores higher than cat and banana because monkeys and bananas are associated through stereotype.
# An example of my own would be "car", "train", "bike". These are all similar as they have wheels and are used for transportation.

# When comparing en_core_web_sm to en_core_web_md the following things can be observed. en_core_web_sm was found to have faster processing speeds but was less accurate than en_core_web_md.
# This is caused by the model being less advanced than en_core_web_md.
