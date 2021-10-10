# Egaroucid - An Othello AI Which Got 10th in The World



## About

**You can play with this AI at [Egaroucid on the Web (in Japanese)](https://www.egaroucid.nyanyan.dev/).**

This AI got 10th in [Codingame Othello](https://www.codingame.com/multiplayer/bot-programming/othello-1/leaderboard) in June, 2021.

You cannot see parameters of this AI, but can see a lot of [game records](https://github.com/Nyanyan/Egaroucid/tree/main/learn/self_play) of this AI.



## How to play

**It's a lot better to play at [Egaroucid on the Web (in Japanese)](https://www.egaroucid.nyanyan.dev/).**

If you really want to play with console,

```
python3 compile.py ai.cpp ai.out
python3 main.py
```



## Technology

I used AlphaZero-like algorithm.

### Deep Learning

This AI uses a neural network which outputs policies and values.

I modified AlphaZero’s network to make it smaller, and I used boards and additional values for its inputs.

### Monte Carlo Tree Search

Almost same as AlphaZero’s.



## What I think and done

Written in Japanese.

Please visit: 

[オセロAI”Egaroucid”全般](https://scrapbox.io/nyanyan/%E3%82%AA%E3%82%BB%E3%83%ADAI%22Egaroucid%22%E5%85%A8%E8%88%AC)

[オセロAI"Egaroucid"におけるAlphaZero風アイデア](https://scrapbox.io/nyanyan/%E3%82%AA%E3%82%BB%E3%83%ADAI%22Egaroucid%22%E3%81%AB%E3%81%8A%E3%81%91%E3%82%8BAlphaZero%E9%A2%A8%E3%82%A2%E3%82%A4%E3%83%87%E3%82%A2)

[オセロAI"Egaroucid"における深層学習の利用](https://scrapbox.io/nyanyan/%E3%82%AA%E3%82%BB%E3%83%ADAI%22Egaroucid%22%E3%81%AB%E3%81%8A%E3%81%91%E3%82%8B%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92%E3%81%AE%E5%88%A9%E7%94%A8)

