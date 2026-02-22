# Retrieval Evaluation Report

## Summary Metrics

| Metric | Value |
|--------|-------|
| Recall@3 | 63.3% |
| Recall@5 | 66.7% |
| Recall@10 | 70.0% |
| MRR (Mean Reciprocal Rank) | 0.516 |
| Number of Questions | 30 |

## Per-Question Results

| # | Question | Difficulty | Hit@3 | Hit@5 | RR | Top Doc |
|---|----------|-----------|-------|-------|----|---------|
| 1 | What does Rule 605 require market centers to discl... | easy | Y | Y | 1.00 | SEC Rule 605 Final Rule (2024) |
| 2 | What is the definition of a covered order under Ru... | easy | Y | Y | 0.33 | SEC Rule 606 Final Rule (2018) |
| 3 | What does FINRA Rule 5310 require of member firms? | easy | N | N | 0.00 | FINRA Regulatory Notice 21-23 |
| 4 | What is interpositioning under FINRA Rule 5310? | easy | N | N | 0.00 | FINRA Regulatory Notice 15-46 |
| 5 | What information must broker-dealers report under ... | easy | Y | Y | 1.00 | SEC Rule 606 Risk Alert |
| 6 | What is the purpose of Regulation NMS? | easy | Y | Y | 0.50 | SEC Regulation NMS Rule 611 Me |
| 7 | How often must Rule 606 reports be published? | easy | N | N | 0.00 | SEC Rule 606 Final Rule (2018) |
| 8 | What factors should firms consider when evaluating... | easy | Y | Y | 1.00 | FINRA Regulatory Notice 21-23 |
| 9 | What is the Order Protection Rule? | easy | Y | Y | 1.00 | SEC Regulation NMS Rule 611 Me |
| 10 | What are the SEC's concerns with Rule 606 complian... | easy | Y | Y | 1.00 | SEC Rule 606 Risk Alert |
| 11 | How did the 2024 amendments change Rule 605 report... | medium | Y | Y | 1.00 | SEC Rule 605 Final Rule (2024) |
| 12 | What constitutes a regular and rigorous review of ... | medium | Y | Y | 1.00 | FINRA Regulatory Notice 15-46 |
| 13 | How does Rule 606 address payment for order flow d... | medium | N | N | 0.12 | FINRA Regulatory Notice 21-23 |
| 14 | What obligations apply when a firm routes orders t... | medium | N | N | 0.00 | SEC Rule 606 Final Rule (2018) |
| 15 | How does Regulation NMS protect investors from tra... | medium | N | Y | 0.20 | SEC Regulation NMS Rule 611 Me |
| 16 | What execution quality metrics must be disclosed u... | medium | Y | Y | 0.50 | SEC Rule 605 Final Rule (2024) |
| 17 | What role does the size of an order play in best e... | medium | Y | Y | 1.00 | FINRA Regulatory Notice 21-23 |
| 18 | How are limit orders treated differently under Rul... | medium | Y | Y | 0.33 | SEC Rule 606 Final Rule (2018) |
| 19 | What are the exceptions to the Order Protection Ru... | medium | N | N | 0.00 | SEC Regulation NMS Rule 611 Me |
| 20 | What documentation must firms maintain for best ex... | medium | Y | Y | 1.00 | FINRA Regulatory Notice 15-46 |
| 21 | How does Rule 606 differ for held versus not-held ... | medium | Y | Y | 1.00 | SEC Rule 606 Final Rule (2018) |
| 22 | What competitive concerns did the SEC identify in ... | medium | Y | Y | 1.00 | SEC Regulation NMS Rule 611 Me |
| 23 | How do Rule 605 and Rule 606 work together to prov... | hard | Y | Y | 0.50 | SEC Rule 606 Final Rule (2018) |
| 24 | What is the relationship between FINRA Rule 5310 a... | hard | N | N | 0.00 | SEC Regulation NMS Rule 611 Me |
| 25 | How should a firm evaluate whether payment for ord... | hard | Y | Y | 1.00 | FINRA Regulatory Notice 21-23 |
| 26 | What options does a firm have when no single marke... | hard | N | N | 0.00 | FINRA Rule 5310 - Best Executi |
| 27 | How do the 2024 Rule 605 amendments address concer... | hard | Y | Y | 0.50 | SEC Rule 605 Fact Sheet |
| 28 | What enforcement actions has FINRA taken related t... | hard | N | N | 0.00 | FINRA Regulatory Notice 15-46 |
| 29 | How does the access fee cap in Rule 610 interact w... | hard | N | N | 0.00 | SEC Rule 605 Final Rule (2024) |
| 30 | What role do self-regulatory organizations play un... | hard | Y | Y | 0.50 | SEC Regulation NMS Rule 611 Me |

## Breakdown by Difficulty

- **Easy** (n=10): Recall@3=70.0%, Recall@5=70.0%, MRR=0.583
- **Medium** (n=12): Recall@3=66.7%, Recall@5=75.0%, MRR=0.597
- **Hard** (n=8): Recall@3=50.0%, Recall@5=50.0%, MRR=0.312

## Failure Analysis

Questions where no relevant result appeared in top-10:

- **What does FINRA Rule 5310 require of member firms?** (difficulty: easy)
- **What is interpositioning under FINRA Rule 5310?** (difficulty: easy)
- **How often must Rule 606 reports be published?** (difficulty: easy)
- **What obligations apply when a firm routes orders to another broker-dealer?** (difficulty: medium)
- **What are the exceptions to the Order Protection Rule?** (difficulty: medium)
- **What is the relationship between FINRA Rule 5310 and SEC Regulation NMS?** (difficulty: hard)
- **What options does a firm have when no single market provides best execution?** (difficulty: hard)
- **What enforcement actions has FINRA taken related to best execution failures?** (difficulty: hard)
- **How does the access fee cap in Rule 610 interact with execution quality analysis?** (difficulty: hard)