# Bioinformatics - Naive Exact Pattern Matching GPU Accelerated
This program can do parallel naive pattern matching in a DNA/RNA sequence.
### *This program only calculates occurrence of a pattern and doesn't store the positions*
## TODO
### 1. Dynamic block initialization in the device
### 2. Store position of every occurrence  

# How to use
The program is built for Nvidia Tesla V100 graphics card. Change numbers of dimensions(blocks, threads) accordingly.

Input is a .txt file without newlines

It has commented code that does the removal of newlines.
```bash
./program_name <pattern>
