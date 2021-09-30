# Analysing Product Reviews

[Saumya et al.](https://arxiv.org/abs/1901.06274) ranked amazon’s reviews using a helpfulness score. The helpfulness score is calculated by using a threshold from each feature extracted from review text data, product description and customer question-answer data of a product. Then the helpfulness score is converted to a binary variable. If it is greater than the threshold then we label the review as high quality, else the review is labelled as low quality.

We will test the proposed method with a public amazon e-commerce sample data. 
