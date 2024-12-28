# 24 Dec
I will begin by checking the logs `train-hmm-vary-100-*` and `train-hmm-vary-e-200-*`.

All the `train-hmm-vary-100-*` was shut down except the first one with index 0. Here is its output

```
1 -34309313.13992910             +nan
Time taken: 14148.878897428513
```

This was in seconds, so roughly 3.9 hours.

The rest was cancelled at `2024-12-20T17:31:50`

For `train-hmm-vary-e-200-*`, we have the same situtation. All was shut down except the first one with the output

```
1 -108862257.88152701             +nan
Time taken: 9959.718718767166
```

This was roughly 2.8 hours.

The `train-hmm-vary-100-*` was to get the timing for one iteration of the model with hidden state size as hs = [200, 400, 600, 800]. Emission was fixed at 100. Context length was fixed at 100.


The `train-hmm-vary-e-200-*` was to do the same thing but the emission is instead fixed at 200. Context length was again fixed at 100. Hidden sizes are varied at [100, 200, 400, 600, 800].

For more comparison, `train_hmm_explore_100` gave a runtime of 1 hours per iter, with model specs L=100, e=100, h=100. It seems like doubling the e will triple the runtime, while doubling the h will quadruple it.