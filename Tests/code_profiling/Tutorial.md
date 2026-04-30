### Installing required programs

```bash
$ pip install memory_profiler
$ pip install line_profiler
$ pip install snakeviz
```

### Line by line profiling:

Add @profile decorators to function you want to profile. Then,

For memory profiling:
```bash
$ python -m memory_profiler pyEFPE_minimal_example.py
```

For time profiling:
```bash
$ kernprof -l -v pyEFPE_minimal_example.py
```

### Whole program time profiling:

```bash
$ python3 -m cProfile -o output.prof pyEFPE_minimal_example.py
$ snakeviz output.prof
```

### Whole program memory profiling:

```bash
$ mprof run --interval 0.05 pyEFPE_minimal_example.py
$ mprof plot
```
