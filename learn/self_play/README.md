# Game Records of Egaroucid

```records/record[N].txt``` is the records of Egaroucid self-play for training.

You can use it for any purpose but I have no responsibility.



## How to Read

They are very text files and you can open it with an editor.

Each line shows each game, such as

```
f5d6c4f6f7d3c3f4c6e7e3c5e8f3b5d2g4d8c7f8f2b3d1b6a7e2f5b4a5g3f1g5a2a3h6a6h3e1c1c2h4a1a4g2c8b1g6h5b7h7b2 -1
```

This record says that

```
black: f5
white: d6
black: c4
...
```

Records don’t have the information about which player to play, but you can guess it with your simulator.

All records start with ```f5``` move.

~~These records end in 51 moves because I can read the result of the game when both player do the best.~~

following number -1, 0, or 1 shows the result of the game. Each number shows:

```
1: black (play the first move (f5)) won
0: draw
-1: white won
```



## License

MIT license, same as this repository



## Contact

If you have questions, contact me!

Twitter: https://twitter.com/Nyanyan_Cube or https://twitter.com/takuto_yamana



