# Analysing Product Reviews

[Saumya et al.](https://arxiv.org/abs/1901.06274) ranked amazon’s reviews using a helpfulness score. The helpfulness score is calculated by using a threshold from each feature extracted from review text data, product description and customer question-answer data of a product. Then the helpfulness score is converted to a binary variable. If it is greater than the threshold then we label the review as high quality, else the review is labelled as low quality.

We will test the proposed method with a public amazon e-commerce sample data. 

## Features
A detailed description of each feature is provided in the Table below.

| Features | Description |
| -------- | --------------------------------------------------------------------------------------------------- |
| Noun     | Number of nouns in the review text |
| Adjective | Number of adjectives in the review text |
| VerB | Number of verbs in the review text |
| Flesch_reading_ease | The Flesch Reading Ease Score. The following scale is helpful in assessing the ease of readability in a document: 90-100: Very Easy, 80-89: Easy, 70-79: Fairly Easy, 60-69: Standard, 50-59: Fairly Difficult, 30-49: Difficult, 0-29: Very Confusing |
| Dale_chall_RE | The DaleChall readability formula is a readability test that provides a numeric gauge of the comprehension difficulty that readers encounter when reading a text. It uses a list of 3000 words that groups of fourth-grade American students could reliably understand, considering any word not on that list to be difficult |
| Difficult_words | Difficult words do not belong to the list of 3000 familiar words |
| Length | Total words in the review |
| Set_length | Total unique words in the review |
| Wrong_words | Words that are not found in Enchant English dictionary |
| One_letter_words | The number of one-letter words in the review |
| Two_letter_words |The number of two-letter words in the review |
| Longer_letter_words | The number of more than two-letter words in the review |
| Lex_diversity | The ratio of unique words to total words in the review |
| Entropy | The entropy indicates how much information is produced on average for each word in the text|
| Rating | The star rating of the product |

